"""
PoseCheck — AIポージング解析 for 競技者
v1.0 MVP

MediaPipe Tasks API (mp.tasks) + Streamlit
全カテゴリ規定ポーズ対応 / 5モード / ダークテーマ
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# ============================================================
# モデルパス（ローカル / Streamlit Cloud 両対応）
# ============================================================
LOCAL_MODEL = "C:/tmp/pose-analyzer/pose_landmarker_heavy.task"
CLOUD_MODEL = str(Path(__file__).parent / "pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

def get_model_path():
    """モデルファイルのパスを取得。なければ自動ダウンロード"""
    if os.path.exists(LOCAL_MODEL):
        return LOCAL_MODEL
    if os.path.exists(CLOUD_MODEL):
        return CLOUD_MODEL
    # Cloud環境: 自動ダウンロード
    import urllib.request
    os.makedirs(os.path.dirname(CLOUD_MODEL) or ".", exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, CLOUD_MODEL)
    return CLOUD_MODEL

MODEL_PATH = get_model_path()

# ============================================================
# ランドマークインデックス
# ============================================================
class LM:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# ============================================================
# 関節角度の定義
# ============================================================
JOINT_ANGLES = {
    "左肘": (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
    "右肘": (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    "左肩": (LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP),
    "右肩": (LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    "左膝": (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
    "右膝": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    "左股関節": (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE),
    "右股関節": (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE),
}

SYMMETRY_PAIRS = [
    ("左肘", "右肘"),
    ("左肩", "右肩"),
    ("左膝", "右膝"),
    ("左股関節", "右股関節"),
]

# 骨格接続（描画用）
POSE_CONNECTIONS = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]

# ============================================================
# 全カテゴリ規定ポーズ定義
# ============================================================

# --- 男子ボディビル ---
BB_FRONT_DOUBLE_BI = {
    "tips": {
        "左肘": {"ideal_range": (60, 90), "instruction": "肘を90度前後に曲げてバイセップスのピークを作る"},
        "右肘": {"ideal_range": (60, 90), "instruction": "肘を90度前後に曲げてバイセップスのピークを作る"},
        "左肩": {"ideal_range": (80, 110), "instruction": "肩を外転させ、腕を水平よりやや上に保つ"},
        "右肩": {"ideal_range": (80, 110), "instruction": "肩を外転させ、腕を水平よりやや上に保つ"},
    },
    "全体": "肩幅を最大限に見せ、ウエストを絞ってXフレームを意識。脚は片足を少し前に出してクアッドの張り出しを見せる。拳は握り込んで二頭筋のピークを高くする",
}

BB_FRONT_LAT = {
    "tips": {
        "左肩": {"ideal_range": (40, 70), "instruction": "肘を腰骨に当て、肩甲骨を外に広げて広背筋を展開する"},
        "右肩": {"ideal_range": (40, 70), "instruction": "肘を腰骨に当て、肩甲骨を外に広げて広背筋を展開する"},
    },
    "全体": "肘を腰骨に固定し、胸を張って広背筋を最大限に広げる。ウエストはバキュームで引き込み、V字シルエットを強調する",
}

BB_SIDE_CHEST = {
    "tips": {
        "左肘": {"ideal_range": (70, 100), "instruction": "審査員側の手首を掴み、胸を押し出すように力を入れる"},
        "右肘": {"ideal_range": (70, 100), "instruction": "審査員側の手首を掴み、胸を押し出すように力を入れる"},
        "左膝": {"ideal_range": (140, 170), "instruction": "前脚の膝を軽く曲げてカーフを浮かせる"},
        "右膝": {"ideal_range": (140, 170), "instruction": "前脚の膝を軽く曲げてカーフを浮かせる"},
    },
    "全体": "審査員側の脚を前に出し、カーフをフレックスして見せる。胸を横から最大限に厚く見せ、腹は引く。肩を上げすぎず自然なラインを保つ",
}

BB_SIDE_TRI = {
    "tips": {
        "左肘": {"ideal_range": (150, 180), "instruction": "腕を後ろで伸ばし、三頭筋の馬蹄形を最大限に見せる"},
        "右肘": {"ideal_range": (150, 180), "instruction": "腕を後ろで伸ばし、三頭筋の馬蹄形を最大限に見せる"},
    },
    "全体": "背中側で手を組み、三頭筋を最大限に収縮させる。サイドチェスト同様に前脚でカーフを見せ、胸の厚みも強調する",
}

BB_BACK_DOUBLE_BI = {
    "tips": {
        "左肘": {"ideal_range": (60, 90), "instruction": "フロントと同様に90度前後でピークを作る"},
        "右肘": {"ideal_range": (60, 90), "instruction": "フロントと同様に90度前後でピークを作る"},
        "左肩": {"ideal_range": (80, 110), "instruction": "広背筋の広がりが見えるように肩を十分に開く"},
        "右肩": {"ideal_range": (80, 110), "instruction": "広背筋の広がりが見えるように肩を十分に開く"},
    },
    "全体": "背中の広がりと厚みを見せる。ハムストリングスとカーフも重要。片足を後ろに引いてカーフをフレックスし、臀筋の分離も見せる",
}

BB_BACK_LAT = {
    "tips": {
        "左肩": {"ideal_range": (40, 70), "instruction": "肘を腰に当て、背中全体を横に広げる"},
        "右肩": {"ideal_range": (40, 70), "instruction": "肘を腰に当て、背中全体を横に広げる"},
    },
    "全体": "肘を腰に固定し、広背筋を最大限に展開。クリスマスツリー（脊柱起立筋の分離）を見せる。脚はバックダブルバイと同様のポジション",
}

BB_ABS_THIGH = {
    "tips": {
        "左肘": {"ideal_range": (80, 120), "instruction": "頭の後ろで手を組み、腹直筋をクランチさせる"},
        "右肘": {"ideal_range": (80, 120), "instruction": "頭の後ろで手を組み、腹直筋をクランチさせる"},
        "左膝": {"ideal_range": (150, 175), "instruction": "片足を前に出し、クアッドのセパレーションを強調"},
        "右膝": {"ideal_range": (150, 175), "instruction": "片足を前に出し、クアッドのセパレーションを強調"},
    },
    "全体": "頭の後ろで手を組み、腹筋を収縮させつつ片足を前に出してクアッドを見せる。バキュームではなくクランチでシックスパックの分離を見せるポーズ",
}

BB_MOST_MUSCULAR = {
    "tips": {
        "左肘": {"ideal_range": (90, 140), "instruction": "両手を前で合わせ、胸・肩・僧帽筋を全収縮させる"},
        "右肘": {"ideal_range": (90, 140), "instruction": "両手を前で合わせ、胸・肩・僧帽筋を全収縮させる"},
        "左肩": {"ideal_range": (30, 70), "instruction": "肩を内旋させて僧帽筋の盛り上がりを強調"},
        "右肩": {"ideal_range": (30, 70), "instruction": "肩を内旋させて僧帽筋の盛り上がりを強調"},
    },
    "全体": "全身の筋肉を最大限に収縮させるパワーポーズ。ハンズオンニー型（片膝に手）かクラブグリップ型（両手を前で組む）を選ぶ。首を短く見せないよう僧帽筋の見せ方に注意",
}

# --- 男子クラシックフィジーク ---
CP_FRONT = {
    "tips": {
        "左肩": {"ideal_range": (50, 90), "instruction": "フロントポーズでは肩のラインを美しく保ち、バキュームでウエストを絞る"},
        "右肩": {"ideal_range": (50, 90), "instruction": "肩幅を見せつつクラシカルなラインを意識する"},
    },
    "全体": "クラシックフィジークはプロポーション重視。バキュームポーズでウエストの細さを強調し、肩幅とのコントラストでVテーパーを見せる。脚は自然に立ち、片足をやや前に",
}

CP_SIDE = {
    "tips": {
        "左肘": {"ideal_range": (70, 110), "instruction": "胸の厚みとアームラインのバランスを意識する"},
        "右肘": {"ideal_range": (70, 110), "instruction": "胸の厚みとアームラインのバランスを意識する"},
    },
    "全体": "クラシカルなサイドポーズ。サイドチェストに近いが、より流れるようなラインを重視。筋量よりプロポーションとポージングの美しさが評価される",
}

CP_BACK = {
    "tips": {
        "左肩": {"ideal_range": (60, 100), "instruction": "背中の広がりとVテーパーを見せるポジションを取る"},
        "右肩": {"ideal_range": (60, 100), "instruction": "背中の広がりとVテーパーを見せるポジションを取る"},
    },
    "全体": "バックポーズでは広背筋の広がりと臀部からハムストリングスのラインを見せる。クラシックらしい美しいシルエットを意識する",
}

CP_FAVOURITE = {
    "tips": {},
    "全体": "自分の最も得意なクラシックポーズを選ぶ。フランクゼイン的なバキュームポーズや、アーノルド的なダブルバイなど。自分のベストパーツを最大限に見せる構図を選択する",
}

# --- メンズフィジーク ---
MP_FRONT = {
    "tips": {
        "左肩": {"ideal_range": (20, 50), "instruction": "腕は自然に下ろし、肩のキャップを見せる。力み過ぎない"},
        "右肩": {"ideal_range": (20, 50), "instruction": "腕は自然に下ろし、肩のキャップを見せる。力み過ぎない"},
    },
    "全体": "リラックスした自然体で立つ。Vテーパー・肩のキャップ・ウエストの細さが評価ポイント。ボードショーツなので脚は見えない。表情は自信を持って笑顔で",
}

MP_RIGHT_SIDE = {
    "tips": {
        "右肩": {"ideal_range": (15, 45), "instruction": "審査員に右半身を見せる。肩のラインを強調しつつリラックス"},
    },
    "全体": "右半身を審査員に向ける。肩の丸みと腕の太さ、ウエストの細さを見せるサイドポーズ。背筋を伸ばし、胸を張る",
}

MP_BACK = {
    "tips": {
        "左肩": {"ideal_range": (20, 50), "instruction": "背中のVテーパーが映えるよう肩を自然に開く"},
        "右肩": {"ideal_range": (20, 50), "instruction": "背中のVテーパーが映えるよう肩を自然に開く"},
    },
    "全体": "バックポーズ。広背筋のVテーパーと肩の広さを見せる。力みすぎず自然なポージングを保つ。ハムストリングスは見えないがポスチャーは重要",
}

MP_LEFT_SIDE = {
    "tips": {
        "左肩": {"ideal_range": (15, 45), "instruction": "審査員に左半身を見せる。肩のラインを強調しつつリラックス"},
    },
    "全体": "左半身を審査員に向ける。右サイドと同様、肩の丸みとウエストのコントラストを見せる",
}

# --- ウィメンズフィジーク（男子BBとほぼ同じ規定） ---
WP_FRONT_DOUBLE_BI = {**BB_FRONT_DOUBLE_BI, "全体": "肩幅を最大限に見せ、ウエストを絞る。女性らしさも意識しつつ筋量をアピール。脚は片足を少し前に出してクアッドを見せる"}
WP_FRONT_LAT = {**BB_FRONT_LAT, "全体": "肘を腰に固定し、広背筋を広げる。女子フィジークでは筋量と女性らしさのバランスが評価される"}
WP_SIDE_CHEST = {**BB_SIDE_CHEST, "全体": "審査員側の脚を前に出し、カーフを見せる。胸の厚みと腹部の引き込みを強調。ラインの美しさも重要"}
WP_SIDE_TRI = {**BB_SIDE_TRI, "全体": "三頭筋の形と腕全体のバランスを見せる。サイドからのシルエットを美しく保つ"}
WP_BACK_DOUBLE_BI = {**BB_BACK_DOUBLE_BI, "全体": "背中の広がりとコンディショニング。ハムストリングスと臀部のラインも見せる"}
WP_BACK_LAT = {**BB_BACK_LAT, "全体": "広背筋を最大限に広げ、Vテーパーを強調。下背部のディテールも評価対象"}
WP_ABS_THIGH = {**BB_ABS_THIGH, "全体": "腹筋のセパレーションとクアッドの分離を見せる。クランチしつつ体全体のバランスを保つ"}
WP_MOST_MUSCULAR = {**BB_MOST_MUSCULAR, "全体": "全身収縮ポーズ。女子フィジークでは過度な力みは減点対象になりうるため、美しさとパワーのバランスを意識"}

# --- フィギュア（女子） ---
FIG_FRONT = {
    "tips": {
        "左肩": {"ideal_range": (15, 40), "instruction": "腕は体側に自然に添え、肩のキャップと鎖骨ラインを見せる"},
        "右肩": {"ideal_range": (15, 40), "instruction": "腕は体側に自然に添え、肩のキャップと鎖骨ラインを見せる"},
    },
    "全体": "筋量と女性らしさのバランスが重要。片足を前に出し、肩幅とウエストのVテーパーを見せる。表情は自信を持って。筋肉の張りとコンディショニングが評価される",
}

FIG_RIGHT_SIDE = {
    "tips": {
        "右肩": {"ideal_range": (15, 40), "instruction": "右半身を見せる。肩から腰へのラインが美しく流れるように"},
    },
    "全体": "サイドからのシルエット。肩のキャップ、腕のライン、ウエスト、ヒップ、脚の流れが評価される。姿勢を正し、胸を張る",
}

FIG_BACK = {
    "tips": {
        "左肩": {"ideal_range": (15, 40), "instruction": "背中のVテーパーが映えるよう自然に立つ"},
        "右肩": {"ideal_range": (15, 40), "instruction": "背中のVテーパーが映えるよう自然に立つ"},
    },
    "全体": "背中のVテーパー、肩の広さ、臀部からハムストリングスのライン。ヒールを履いた状態でのカーフの発達も評価対象",
}

FIG_LEFT_SIDE = {
    "tips": {
        "左肩": {"ideal_range": (15, 40), "instruction": "左半身を見せる。右サイドと対称的なシルエットを意識"},
    },
    "全体": "右サイドの鏡像。左半身からのシルエット美を見せる",
}

# --- ビキニ（女子） ---
BK_FRONT = {
    "tips": {
        "左肩": {"ideal_range": (10, 35), "instruction": "腕はリラックスして体側に。肩のラインと全体のバランスを見せる"},
        "右肩": {"ideal_range": (10, 35), "instruction": "腕はリラックスして体側に。肩のラインと全体のバランスを見せる"},
        "左膝": {"ideal_range": (160, 180), "instruction": "脚はまっすぐ伸ばし、片足を少し前にずらす"},
        "右膝": {"ideal_range": (160, 180), "instruction": "脚はまっすぐ伸ばし、片足を少し前にずらす"},
    },
    "全体": "バランスのとれたプロポーションと適度な筋肉の張り。ステージプレゼンス、自信、笑顔が重要。ハイヒールでの立ち姿、S字カーブを意識。過度な筋量はマイナス",
}

BK_BACK = {
    "tips": {
        "左肩": {"ideal_range": (10, 35), "instruction": "背中側のラインを美しく見せる。肩を落としすぎない"},
        "右肩": {"ideal_range": (10, 35), "instruction": "背中側のラインを美しく見せる。肩を落としすぎない"},
    },
    "全体": "バックポーズ。臀部の形、ハムストリングスとのつながり、背中のライン。片足を後ろに引いてヒップラインを強調。グルートのコンディショニングが評価される",
}

# --- ウェルネス（女子） ---
WL_FRONT = {
    "tips": {
        "左肩": {"ideal_range": (10, 35), "instruction": "上半身はビキニに近い自然体。下半身のボリュームとのバランスを見せる"},
        "右肩": {"ideal_range": (10, 35), "instruction": "上半身はビキニに近い自然体。下半身のボリュームとのバランスを見せる"},
    },
    "全体": "下半身（臀部・脚）のボリュームと上半身のバランスが最大の評価軸。ビキニより下半身の筋量が多く、そのプロポーションを見せる。片足を前に出してクアッドの丸みを強調",
}

WL_RIGHT_SIDE = {
    "tips": {
        "右肩": {"ideal_range": (10, 35), "instruction": "横からのシルエット。臀部のボリュームと背中のラインを見せる"},
    },
    "全体": "サイドポーズでは臀部の突き出しとハムストリングスからヒップへのカーブが評価される。ウエストの細さとのコントラストも重要",
}

WL_BACK = {
    "tips": {
        "左肩": {"ideal_range": (10, 35), "instruction": "背中側から臀部と脚のボリュームを見せる"},
        "右肩": {"ideal_range": (10, 35), "instruction": "背中側から臀部と脚のボリュームを見せる"},
    },
    "全体": "ウェルネスの最大の見せ場。グルートのボリューム、丸み、ハムとのつながり。片足を引いてヒップラインを最大限に強調する",
}

WL_LEFT_SIDE = {
    "tips": {
        "左肩": {"ideal_range": (10, 35), "instruction": "左サイドからのシルエット。右サイドと同様のラインを意識"},
    },
    "全体": "右サイドの鏡像。左半身からの臀部とウエストのコントラストを見せる",
}


# カテゴリ → ポーズ名 → 定義 のマスターデータ
POSE_CATEGORIES = {
    "男子ボディビル": {
        "フロントダブルバイセップス": BB_FRONT_DOUBLE_BI,
        "フロントラットスプレッド": BB_FRONT_LAT,
        "サイドチェスト（左右）": BB_SIDE_CHEST,
        "サイドトライセップス（左右）": BB_SIDE_TRI,
        "バックダブルバイセップス": BB_BACK_DOUBLE_BI,
        "バックラットスプレッド": BB_BACK_LAT,
        "アブドミナル&サイ": BB_ABS_THIGH,
        "モストマスキュラー": BB_MOST_MUSCULAR,
    },
    "男子クラシックフィジーク": {
        "フロントポーズ（バキューム含む）": CP_FRONT,
        "サイドポーズ": CP_SIDE,
        "バックポーズ": CP_BACK,
        "フェイバリットクラシックポーズ": CP_FAVOURITE,
    },
    "メンズフィジーク": {
        "フロント": MP_FRONT,
        "右サイド": MP_RIGHT_SIDE,
        "バック": MP_BACK,
        "左サイド": MP_LEFT_SIDE,
    },
    "ウィメンズフィジーク": {
        "フロントダブルバイセップス": WP_FRONT_DOUBLE_BI,
        "フロントラットスプレッド": WP_FRONT_LAT,
        "サイドチェスト（左右）": WP_SIDE_CHEST,
        "サイドトライセップス（左右）": WP_SIDE_TRI,
        "バックダブルバイセップス": WP_BACK_DOUBLE_BI,
        "バックラットスプレッド": WP_BACK_LAT,
        "アブドミナル&サイ": WP_ABS_THIGH,
        "モストマスキュラー": WP_MOST_MUSCULAR,
    },
    "フィギュア（女子）": {
        "フロント": FIG_FRONT,
        "右サイド": FIG_RIGHT_SIDE,
        "バック": FIG_BACK,
        "左サイド": FIG_LEFT_SIDE,
    },
    "ビキニ（女子）": {
        "フロントポーズ": BK_FRONT,
        "バックポーズ": BK_BACK,
    },
    "ウェルネス（女子）": {
        "フロント": WL_FRONT,
        "右サイド": WL_RIGHT_SIDE,
        "バック": WL_BACK,
        "左サイド": WL_LEFT_SIDE,
    },
}


# ============================================================
# コア解析関数
# ============================================================

def calc_angle(a, b, c):
    """3点 a-b-c から b を頂点とする角度を計算（度数法）"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def draw_landmarks(image, landmarks):
    """ランドマークと骨格接続を画像に描画"""
    h, w = image.shape[:2]
    annotated = image.copy()

    # 接続線（ティール系）
    for p1_idx, p2_idx in POSE_CONNECTIONS:
        if p1_idx < len(landmarks) and p2_idx < len(landmarks):
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(annotated, (x1, y1), (x2, y2), (180, 230, 20), 2)

    # 関節点（ティール）
    key_indices = {LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
                   LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_HIP, LM.RIGHT_HIP,
                   LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_ANKLE, LM.RIGHT_ANKLE}
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        if i in key_indices:
            cv2.circle(annotated, (x, y), 6, (0, 200, 200), -1)
            cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1)
        else:
            cv2.circle(annotated, (x, y), 3, (100, 100, 100), -1)

    return annotated


def get_angles_from_landmarks(landmarks):
    """ランドマークリストから全関節角度を計算"""
    angles = {}
    for name, (p1, p2, p3) in JOINT_ANGLES.items():
        try:
            a = [landmarks[p1].x, landmarks[p1].y]
            b = [landmarks[p2].x, landmarks[p2].y]
            c = [landmarks[p3].x, landmarks[p3].y]
            angles[name] = round(calc_angle(a, b, c), 1)
        except (IndexError, Exception):
            angles[name] = None
    return angles


def calc_symmetry_score(angles):
    """左右対称性スコア（100点満点）"""
    diffs = []
    details = []
    for left, right in SYMMETRY_PAIRS:
        if angles.get(left) is not None and angles.get(right) is not None:
            diff = abs(angles[left] - angles[right])
            diffs.append(diff)
            details.append({
                "部位": left.replace("左", ""),
                "左": angles[left],
                "右": angles[right],
                "差": round(diff, 1),
            })

    if not diffs:
        return None, details

    avg_diff = np.mean(diffs)
    score = max(0, 100 - (avg_diff / 30 * 100))
    return round(score, 1), details


def create_detector(is_video=False):
    """PoseLandmarkerを作成"""
    if is_video:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
    return vision.PoseLandmarker.create_from_options(options)


def analyze_image(image_path):
    """静止画解析 → (annotated_bgr, angles) or (None, None)"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )

    detector = create_detector(is_video=False)
    result = detector.detect(mp_image)
    detector.close()

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None, None

    landmarks = result.pose_landmarks[0]
    angles = get_angles_from_landmarks(landmarks)
    annotated = draw_landmarks(image, landmarks)

    return annotated, angles


def analyze_video(video_path, max_frames=900, target_fps=5):
    """動画解析 → フレームデータのリスト"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_rate = max(1, int(fps / target_fps))

    detector = create_detector(is_video=True)

    frame_data = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int(frame_count / fps * 1000)

            try:
                result = detector.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    angles = get_angles_from_landmarks(landmarks)
                    score, details = calc_symmetry_score(angles)
                    annotated = draw_landmarks(frame, landmarks)
                    time_sec = round(frame_count / fps, 1)

                    frame_data.append({
                        "frame_idx": frame_count,
                        "time": time_sec,
                        "angles": angles,
                        "score": score,
                        "details": details,
                        "annotated": annotated,
                    })
            except Exception:
                pass

        frame_count += 1

    detector.close()
    cap.release()
    return frame_data


# ============================================================
# フィードバック生成
# ============================================================

def generate_body_instruction(joint_name, diff, me_val, ref_val):
    """関節名と差分から、身体の具体的な動かし方を日本語で生成"""
    abs_diff = abs(diff)
    direction = "閉じる" if diff > 0 else "開く"

    instructions = {
        "左肘": {
            "閉じる": f"左肘をあと{abs_diff:.0f}度曲げる。前腕を上腕に近づけるイメージ",
            "開く": f"左肘をあと{abs_diff:.0f}度伸ばす。力を入れつつ腕を開くイメージ",
        },
        "右肘": {
            "閉じる": f"右肘をあと{abs_diff:.0f}度曲げる。前腕を上腕に近づけるイメージ",
            "開く": f"右肘をあと{abs_diff:.0f}度伸ばす。力を入れつつ腕を開くイメージ",
        },
        "左肩": {
            "閉じる": f"左腕を体側に{abs_diff:.0f}度近づける。脇を締めるイメージ",
            "開く": f"左腕をあと{abs_diff:.0f}度外に開く。肩甲骨を寄せるか外転させる",
        },
        "右肩": {
            "閉じる": f"右腕を体側に{abs_diff:.0f}度近づける。脇を締めるイメージ",
            "開く": f"右腕をあと{abs_diff:.0f}度外に開く。肩甲骨を寄せるか外転させる",
        },
        "左膝": {
            "閉じる": f"左膝をあと{abs_diff:.0f}度曲げる。やや腰を落とすイメージ",
            "開く": f"左膝をあと{abs_diff:.0f}度伸ばす。脚をまっすぐに近づける",
        },
        "右膝": {
            "閉じる": f"右膝をあと{abs_diff:.0f}度曲げる。やや腰を落とすイメージ",
            "開く": f"右膝をあと{abs_diff:.0f}度伸ばす。脚をまっすぐに近づける",
        },
        "左股関節": {
            "閉じる": f"上体を左側にあと{abs_diff:.0f}度かがめる。もしくは左脚を後ろに引く",
            "開く": f"上体を起こし、左股関節をあと{abs_diff:.0f}度開く",
        },
        "右股関節": {
            "閉じる": f"上体を右側にあと{abs_diff:.0f}度かがめる。もしくは右脚を後ろに引く",
            "開く": f"上体を起こし、右股関節をあと{abs_diff:.0f}度開く",
        },
    }

    default = f"あと{abs_diff:.0f}度{direction}"
    return instructions.get(joint_name, {}).get(direction, default)


def generate_feedback(angles_ref, angles_me, pose_info):
    """お手本と自分の角度差分からフィードバックリストを生成"""
    feedback_items = []

    for name in JOINT_ANGLES:
        ref_val = angles_ref.get(name)
        me_val = angles_me.get(name)
        if ref_val is None or me_val is None:
            continue

        diff = me_val - ref_val
        abs_diff = abs(diff)

        if abs_diff < 5:
            status = "OK"
            priority = 0
            instruction = "お手本とほぼ一致"
        elif abs_diff < 15:
            status = "WARN"
            priority = 1
            instruction = generate_body_instruction(name, diff, me_val, ref_val)
        else:
            status = "FIX"
            priority = 2
            instruction = generate_body_instruction(name, diff, me_val, ref_val)

        # ポーズ固有のtipsを付加
        extra_tip = ""
        if pose_info and name in pose_info.get("tips", {}):
            tip = pose_info["tips"][name]
            low, high = tip["ideal_range"]
            extra_tip = tip["instruction"]
            if me_val < low or me_val > high:
                extra_tip += f"（現在{me_val:.0f}度 / 理想{low}~{high}度）"

        feedback_items.append({
            "status": status,
            "name": name,
            "ref": ref_val,
            "me": me_val,
            "diff": diff,
            "abs_diff": abs_diff,
            "instruction": instruction,
            "extra_tip": extra_tip,
            "priority": priority,
        })

    feedback_items.sort(key=lambda x: (-x["priority"], -x["abs_diff"]))
    return feedback_items


def calc_match_score(angles_ref, angles_me):
    """お手本との一致度スコア（100点満点）"""
    diffs = []
    for name in JOINT_ANGLES:
        ref_val = angles_ref.get(name)
        me_val = angles_me.get(name)
        if ref_val is not None and me_val is not None:
            diffs.append(abs(ref_val - me_val))
    if not diffs:
        return None
    avg_diff = np.mean(diffs)
    return round(max(0, 100 - (avg_diff / 30 * 100)), 1)


# ============================================================
# UI ヘルパー
# ============================================================

def show_symmetry_badge(score):
    """対称性スコアをバッジ表示"""
    if score >= 90:
        st.success(f"対称性: {score} / 100（優秀）")
    elif score >= 75:
        st.warning(f"対称性: {score} / 100（改善の余地あり）")
    else:
        st.error(f"対称性: {score} / 100（要改善）")


def show_match_badge(score):
    """一致度スコアをバッジ表示"""
    if score >= 85:
        st.success(f"一致度: {score} / 100（かなり近い）")
    elif score >= 65:
        st.warning(f"一致度: {score} / 100（調整すればいける）")
    else:
        st.error(f"一致度: {score} / 100（大幅な修正が必要）")


def save_uploaded_file(uploaded, suffix=".jpg"):
    """アップロードファイルを一時保存してパスを返す"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


def display_angles_table(angles):
    """関節角度をメトリクスで表示"""
    cols = st.columns(4)
    items = [(k, v) for k, v in angles.items() if v is not None]
    for i, (name, angle) in enumerate(items):
        with cols[i % 4]:
            st.metric(name, f"{angle}°")


def display_symmetry_details(details):
    """左右差の詳細を表示"""
    for d in details:
        icon = "🟢" if d["差"] < 5 else "🟡" if d["差"] < 10 else "🔴"
        st.text(f"  {icon} {d['部位']}: 左 {d['左']}° / 右 {d['右']}°（差 {d['差']}°）")


# ============================================================
# Streamlit メインUI
# ============================================================

st.set_page_config(
    page_title="PoseCheck — AIポージング解析",
    page_icon="🏋️",
    layout="wide",
)

# --- ヘッダー ---
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0;">
    <h1 style="margin: 0; font-size: 2.2rem; color: #14b8a6;">PoseCheck</h1>
    <p style="margin: 0; color: #94a3b8; font-size: 1rem;">AIポージング解析 for 競技者</p>
</div>
""", unsafe_allow_html=True)

# --- サイドバー ---
st.sidebar.markdown("### モード選択")
mode = st.sidebar.radio(
    "解析モード",
    [
        "📸 静止画解析",
        "🎬 動画解析",
        "🏆 ポージング動画解析",
        "🎯 お手本比較",
        "🔄 比較モード",
    ],
    label_visibility="collapsed",
)

# カテゴリ・ポーズ選択（お手本比較モード用だがサイドバーに常時表示）
st.sidebar.divider()
st.sidebar.markdown("### カテゴリ / ポーズ")
selected_category = st.sidebar.selectbox(
    "カテゴリ",
    list(POSE_CATEGORIES.keys()),
)
pose_names = list(POSE_CATEGORIES[selected_category].keys())
selected_pose = st.sidebar.selectbox(
    "ポーズ",
    pose_names,
)
current_pose_info = POSE_CATEGORIES[selected_category][selected_pose]

# ポーズのコツ表示
with st.sidebar.expander("選択中のポーズのコツ"):
    st.markdown(f"**{selected_category} / {selected_pose}**")
    st.write(current_pose_info.get("全体", ""))
    tips = current_pose_info.get("tips", {})
    if tips:
        st.markdown("**理想角度の目安:**")
        shown = set()
        for joint, tip in tips.items():
            label = joint.replace("左", "").replace("右", "")
            if label not in shown:
                st.text(f"  {label}: {tip['ideal_range'][0]}~{tip['ideal_range'][1]}°")
                shown.add(label)

st.sidebar.divider()
st.sidebar.caption("PoseCheck v1.0 MVP")
st.sidebar.caption("MediaPipe Tasks API + Streamlit")

# --- 使い方ガイド ---
with st.expander("使い方ガイド（初めての方はこちら）"):
    st.markdown("""
**PoseCheck** は骨格検出AIを使ってポージングを数値化・解析するツールです。

**モードの説明:**
| モード | 用途 |
|--------|------|
| 📸 静止画解析 | 1枚の写真から骨格・関節角度・対称性スコアを算出 |
| 🎬 動画解析 | トレーニングフォームの関節角度推移・可動域（ROM）を可視化 |
| 🏆 ポージング動画解析 | ポージング動画全体の対称性スコア推移。ベスト/ワーストフレーム自動抽出 |
| 🎯 お手本比較 | プロのポージング写真と自分を比較。具体的な修正指示を取得 |
| 🔄 比較モード | 2枚のビフォーアフターを並べて角度差分を比較 |

**撮影のコツ:**
- 全身が映るように撮影する（頭からつま先まで）
- 背景はできるだけシンプルに
- 明るい場所で撮影する
- カメラは正面 or 真横から。斜めだと角度がずれる

**サイドバーの使い方:**
1. 左のサイドバーでカテゴリ（男子ボディビル、メンズフィジーク等）を選択
2. 続けてポーズを選択
3. 選択したポーズのコツが表示される
4. お手本比較モードでは、選択したポーズに応じた修正フィードバックが生成される
""")


# ============================================================
# モード: 📸 静止画解析
# ============================================================
if mode == "📸 静止画解析":
    st.header("📸 静止画解析")
    st.caption("1枚の写真から骨格検出・関節角度・対称性スコアを算出します")

    uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded:
        tmp_path = save_uploaded_file(uploaded, ".jpg")
        annotated, angles = analyze_image(tmp_path)
        os.unlink(tmp_path)

        if annotated is not None and angles:
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader("骨格検出結果")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.subheader("関節角度")
                display_angles_table(angles)

                score, details = calc_symmetry_score(angles)
                if score is not None:
                    st.divider()
                    st.subheader("左右対称性")
                    show_symmetry_badge(score)
                    display_symmetry_details(details)

                # ポーズ固有の理想角度との比較
                tips = current_pose_info.get("tips", {})
                if tips:
                    st.divider()
                    st.subheader(f"ポーズチェック: {selected_pose}")
                    for joint, tip in tips.items():
                        val = angles.get(joint)
                        if val is not None:
                            low, high = tip["ideal_range"]
                            if low <= val <= high:
                                st.text(f"  🟢 {joint}: {val}°（理想範囲内: {low}~{high}°）")
                            else:
                                st.text(f"  🔴 {joint}: {val}°（理想: {low}~{high}°）→ {tip['instruction']}")
        else:
            st.error("骨格を検出できませんでした。全身が写っている画像を使用してください。")


# ============================================================
# モード: 🎬 動画解析
# ============================================================
elif mode == "🎬 動画解析":
    st.header("🎬 動画解析")
    st.caption("トレーニングフォームの関節角度推移と可動域（ROM）を分析します")

    uploaded = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])

    if uploaded:
        suffix = "." + (uploaded.name.split(".")[-1] if "." in uploaded.name else "mp4")
        tmp_path = save_uploaded_file(uploaded, suffix)

        with st.spinner("解析中...（動画の長さにより数十秒かかります）"):
            frame_data = analyze_video(tmp_path)

        os.unlink(tmp_path)

        if frame_data:
            st.success(f"解析完了: {len(frame_data)} フレーム（{frame_data[-1]['time']}秒）")

            # サンプルフレーム
            st.subheader("検出フレーム")
            n_show = min(4, len(frame_data))
            if n_show > 1:
                indices = [0] + [len(frame_data) * i // (n_show - 1) for i in range(1, n_show - 1)] + [len(frame_data) - 1]
            else:
                indices = [0]
            indices = sorted(set(indices))

            cols = st.columns(len(indices))
            for i, idx in enumerate(indices):
                d = frame_data[idx]
                with cols[i]:
                    st.image(cv2.cvtColor(d["annotated"], cv2.COLOR_BGR2RGB),
                             caption=f"{d['time']}秒", use_container_width=True)

            # 角度推移グラフ
            st.subheader("関節角度の推移")
            all_angles = [d["angles"] for d in frame_data]
            frame_times = [d["time"] for d in frame_data]
            df = pd.DataFrame(all_angles, index=frame_times)
            df.index.name = "時間（秒）"

            selected_joints = st.multiselect(
                "表示する関節を選択",
                list(JOINT_ANGLES.keys()),
                default=["左膝", "右膝"],
            )
            if selected_joints:
                st.line_chart(df[selected_joints])

            # ROM
            st.subheader("可動域（ROM）")
            rom_data = []
            for name in JOINT_ANGLES:
                vals = [a[name] for a in all_angles if a.get(name) is not None]
                if vals:
                    rom_data.append({
                        "関節": name,
                        "最小": f"{min(vals):.0f}°",
                        "最大": f"{max(vals):.0f}°",
                        "可動域": f"{max(vals) - min(vals):.0f}°",
                    })
            if rom_data:
                st.dataframe(pd.DataFrame(rom_data), use_container_width=True, hide_index=True)
        else:
            st.error("骨格を検出できませんでした。全身が写っている動画を使用してください。")


# ============================================================
# モード: 🏆 ポージング動画解析
# ============================================================
elif mode == "🏆 ポージング動画解析":
    st.header("🏆 ポージング動画解析")
    st.caption("ポージング動画全体の対称性スコア推移を追跡し、ベスト/ワーストフレームを自動抽出します")

    uploaded = st.file_uploader("ポージング動画をアップロード", type=["mp4", "mov", "avi"])

    if uploaded:
        suffix = "." + (uploaded.name.split(".")[-1] if "." in uploaded.name else "mp4")
        tmp_path = save_uploaded_file(uploaded, suffix)

        with st.spinner("ポージング解析中..."):
            frame_data = analyze_video(tmp_path, target_fps=5)

        os.unlink(tmp_path)

        scored_data = [d for d in frame_data if d["score"] is not None]

        if scored_data:
            st.success(f"解析完了: {len(scored_data)} フレーム（{scored_data[-1]['time']}秒）")

            # 対称性スコア推移
            st.subheader("対称性スコア推移")
            score_df = pd.DataFrame({
                "時間（秒）": [d["time"] for d in scored_data],
                "対称性スコア": [d["score"] for d in scored_data],
            }).set_index("時間（秒）")
            st.line_chart(score_df)

            # 統計
            scores = [d["score"] for d in scored_data]
            col1, col2, col3 = st.columns(3)
            col1.metric("平均スコア", f"{np.mean(scores):.1f}")
            col2.metric("最高スコア", f"{max(scores):.1f}")
            col3.metric("最低スコア", f"{min(scores):.1f}")

            st.divider()

            # ベスト & ワースト
            sorted_data = sorted(scored_data, key=lambda x: x["score"], reverse=True)
            best = sorted_data[0]
            worst = sorted_data[-1]

            st.subheader("ベストフレーム vs ワーストフレーム")
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"ベスト: {best['score']}/100（{best['time']}秒）")
                st.image(cv2.cvtColor(best["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
                display_symmetry_details(best["details"])

            with col2:
                st.error(f"ワースト: {worst['score']}/100（{worst['time']}秒）")
                st.image(cv2.cvtColor(worst["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
                display_symmetry_details(worst["details"])

            st.divider()

            # フレームスライダー
            st.subheader("フレーム詳細")
            frame_idx = st.slider("フレームを選択", 0, len(scored_data) - 1, 0)
            sel = scored_data[frame_idx]

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(sel["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"時間: {sel['time']}秒")

            with col2:
                show_symmetry_badge(sel["score"])
                display_angles_table(sel["angles"])
                st.divider()
                st.markdown("**左右差:**")
                display_symmetry_details(sel["details"])

            # 部位別左右差推移
            st.subheader("部位別 左右差の推移")
            diff_data = {}
            for pair_left, pair_right in SYMMETRY_PAIRS:
                part_name = pair_left.replace("左", "")
                diff_data[part_name] = [
                    abs((d["angles"].get(pair_left) or 0) - (d["angles"].get(pair_right) or 0))
                    for d in scored_data
                ]

            diff_df = pd.DataFrame(diff_data, index=[d["time"] for d in scored_data])
            diff_df.index.name = "時間（秒）"
            st.line_chart(diff_df)

        else:
            st.error("骨格を検出できませんでした。全身が写っている動画を使用してください。")


# ============================================================
# モード: 🎯 お手本比較
# ============================================================
elif mode == "🎯 お手本比較":
    st.header("🎯 お手本ポーズとの比較")
    st.caption(f"カテゴリ: **{selected_category}** / ポーズ: **{selected_pose}**（サイドバーで変更可能）")

    # ポーズの全体アドバイスを表示
    st.info(f"**{selected_pose}のポイント:** {current_pose_info.get('全体', '')}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("お手本")
        file_ref = st.file_uploader("お手本画像（プロの写真等）", type=["jpg", "jpeg", "png"], key="ref")

    with col2:
        st.subheader("自分")
        file_me = st.file_uploader("自分のポージング写真", type=["jpg", "jpeg", "png"], key="me")

    if file_ref and file_me:
        # 解析
        tmp_ref = save_uploaded_file(file_ref, ".jpg")
        annotated_ref, angles_ref = analyze_image(tmp_ref)
        os.unlink(tmp_ref)

        tmp_me = save_uploaded_file(file_me, ".jpg")
        annotated_me, angles_me = analyze_image(tmp_me)
        os.unlink(tmp_me)

        if annotated_ref is not None and annotated_me is not None:
            # 画像並列表示
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(annotated_ref, cv2.COLOR_BGR2RGB),
                         caption="お手本", use_container_width=True)
                score_ref, _ = calc_symmetry_score(angles_ref)
                if score_ref:
                    st.metric("対称性スコア", f"{score_ref}/100")

            with col2:
                st.image(cv2.cvtColor(annotated_me, cv2.COLOR_BGR2RGB),
                         caption="自分", use_container_width=True)
                score_me, _ = calc_symmetry_score(angles_me)
                if score_me:
                    st.metric("対称性スコア", f"{score_me}/100")

            st.divider()

            # --- 一致度スコア ---
            match_score = calc_match_score(angles_ref, angles_me)
            if match_score is not None:
                st.subheader("お手本との一致度")
                show_match_badge(match_score)

            st.divider()

            # --- 修正フィードバック ---
            st.subheader("修正フィードバック")

            feedback_items = generate_feedback(angles_ref, angles_me, current_pose_info)

            # 優先修正ポイントTOP3
            top_fixes = [f for f in feedback_items if f["priority"] >= 1][:3]
            if top_fixes:
                st.markdown("#### 優先修正ポイント TOP3")
                for i, f in enumerate(top_fixes, 1):
                    icon = "🔴" if f["priority"] == 2 else "🟡"
                    st.markdown(f"**{i}. {icon} {f['name']}** — {f['instruction']}")
                    if f["extra_tip"]:
                        st.caption(f"    ヒント: {f['extra_tip']}")
                st.divider()

            # 全関節の詳細
            st.markdown("#### 全関節の詳細")
            for item in feedback_items:
                if item["status"] == "OK":
                    icon = "✅"
                elif item["status"] == "WARN":
                    icon = "🟡"
                else:
                    icon = "🔴"

                with st.container():
                    st.markdown(
                        f"{icon} **{item['name']}** — "
                        f"お手本: {item['ref']:.0f}° / 自分: {item['me']:.0f}° "
                        f"（差: {'+' if item['diff'] > 0 else ''}{item['diff']:.0f}°）"
                    )
                    detail_text = f"  → {item['instruction']}"
                    if item["extra_tip"]:
                        detail_text += f" | {item['extra_tip']}"
                    st.caption(detail_text)

            # サマリー
            st.divider()
            n_ok = len([f for f in feedback_items if f["priority"] == 0])
            n_warn = len([f for f in feedback_items if f["priority"] == 1])
            n_fix = len([f for f in feedback_items if f["priority"] == 2])
            st.markdown(
                f"**サマリー**: ✅ OK {n_ok}箇所 / 🟡 微調整 {n_warn}箇所 / 🔴 要修正 {n_fix}箇所"
            )

        else:
            st.error("骨格を検出できませんでした。全身が写っている画像を使用してください。")


# ============================================================
# モード: 🔄 比較モード
# ============================================================
elif mode == "🔄 比較モード":
    st.header("🔄 ビフォーアフター比較")
    st.caption("2枚の画像の骨格・角度を並べて比較します")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("A（ビフォー）")
        file_a = st.file_uploader("画像A", type=["jpg", "jpeg", "png"], key="a")

    with col2:
        st.subheader("B（アフター）")
        file_b = st.file_uploader("画像B", type=["jpg", "jpeg", "png"], key="b")

    if file_a and file_b:
        tmp_a = save_uploaded_file(file_a, ".jpg")
        annotated_a, angles_a = analyze_image(tmp_a)
        os.unlink(tmp_a)

        tmp_b = save_uploaded_file(file_b, ".jpg")
        annotated_b, angles_b = analyze_image(tmp_b)
        os.unlink(tmp_b)

        if annotated_a is not None and annotated_b is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(annotated_a, cv2.COLOR_BGR2RGB),
                         caption="A", use_container_width=True)
                score_a, details_a = calc_symmetry_score(angles_a)
                if score_a:
                    show_symmetry_badge(score_a)

            with col2:
                st.image(cv2.cvtColor(annotated_b, cv2.COLOR_BGR2RGB),
                         caption="B", use_container_width=True)
                score_b, details_b = calc_symmetry_score(angles_b)
                if score_b:
                    show_symmetry_badge(score_b)

            # 角度比較テーブル
            st.divider()
            st.subheader("角度比較")
            comparison = []
            for name in JOINT_ANGLES:
                a_val = angles_a.get(name)
                b_val = angles_b.get(name)
                if a_val is not None and b_val is not None:
                    diff = round(b_val - a_val, 1)
                    comparison.append({
                        "関節": name,
                        "A": f"{a_val}°",
                        "B": f"{b_val}°",
                        "差分（B-A）": f"{'+' if diff > 0 else ''}{diff}°",
                    })
            if comparison:
                st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

            # 対称性比較
            if score_a is not None and score_b is not None:
                st.divider()
                st.subheader("対称性スコア比較")
                diff_score = round(score_b - score_a, 1)
                col1, col2, col3 = st.columns(3)
                col1.metric("A", f"{score_a}/100")
                col2.metric("B", f"{score_b}/100")
                col3.metric("変化", f"{'+' if diff_score > 0 else ''}{diff_score}",
                            delta=f"{'+' if diff_score > 0 else ''}{diff_score}")

        else:
            st.error("骨格を検出できませんでした。全身が写っている画像を使用してください。")
