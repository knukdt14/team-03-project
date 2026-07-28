"""
검색 난이도를 높인 "어려운 버전" 평가셋.

기존 BENCHMARK_DATASET은 매뉴얼 원문의 "기능명(한글명): 설명" 패턴을 그대로 질문화해서,
질문과 정답 문단의 어휘가 1:1로 겹친다 -> 임베딩 검색이 거의 문자열 매칭 수준으로 쉬워지고,
그 결과 chunk_size/top_k/Query Expansion 등 어떤 파라미터를 바꿔도 검색 결과가 안 바뀐다.
(eval/PERFORMANCE_TUNING_SUMMARY.md 참고)

이 세트는 같은 사실을 묻되 **기능명(정답 키워드)을 질문에서 배제하고 동작/목적으로 우회 서술**해서,
질문-문단 사이 어휘 간극을 만든다. 정답이 애매해지지 않도록 단일 기능으로 명확히 떨어지는
문항만 골랐다(비교/멀티홉형은 제외).

ground_truth는 기존 세트와 동일한 매뉴얼 원문 사실을 사용한다.
"""

HARD_BENCHMARK_DATASET = [
    {
        "type": "Hard/Paraphrased QA",
        "original": "Part Design의 Hole 기능은 무엇인가요?",
        "question": "부품에 둥근 구멍을 뚫으려면 어떤 도구를 써야 하나요?",
        "ground_truth": "Hole(홀)은 원형의 구멍을 만드는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Pad, Pocket 기능 중 Pocket의 역할은 무엇인가요?",
        "question": "이미 만든 3차원 덩어리에서 안쪽으로 파내어 빈 공간을 만들고 싶습니다. 어떤 기능인가요?",
        "ground_truth": "Pocket(포켓)은 3차원 형상에서 2차원 형상의 공간을 만들어내는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Shaft 기능은 무엇인가요?",
        "question": "단면을 축 중심으로 빙 돌려서 입체를 만드는 방식은 무엇이라고 하나요?",
        "ground_truth": "Shaft(쉐프트)는 중심축을 기준으로 2차원 단면을 회전시켜 3차원 형상을 만드는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Part Design의 Shell 기능은 언제 사용하나요?",
        "question": "속을 비워서 껍데기만 남기고 벽 두께만 유지하고 싶을 때 뭘 쓰나요?",
        "ground_truth": "Shell(쉘)은 형상의 일정한 두께를 남기고 빈 공간을 만들고자 할 때 사용하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Chamfer 기능은 언제 사용하나요?",
        "question": "날카로운 모서리를 비스듬하게 깎아내고 싶은데 어떻게 하나요?",
        "ground_truth": "Chamfer(모따기)는 면과 면 사이에 경사가 필요한 경우, 모서리에 모따기를 생성할 때 사용합니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Part Design의 Edge Fillet 기능은 언제 사용하나요?",
        "question": "모서리를 둥글둥글하게 부드러운 곡면으로 처리하려면 어떤 기능을 쓰나요?",
        "ground_truth": "Edge Fillet(엣지필렛)은 면과 면 사이에 부드러운 연결 형상이 필요한 경우에 사용하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Draft Angle 기능이란 무엇이며 어떨 때 사용하나요?",
        "question": "사출 금형에서 굳은 부품이 잘 안 빠질 때, 옆면에 기울기를 줘서 빼내기 쉽게 하려면 어떻게 하나요?",
        "ground_truth": "Draft Angle 기능은 금형(Molding) 공정에서 부품을 원활하게 빼내기 위해 측면에 꺾임각/경사각을 부여하는 Dress-Up 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Part Design의 Thickness 기능은 무엇인가요?",
        "question": "이미 돌출시켜 만든 형상의 벽 두께 값을 나중에 바꾸고 싶으면 어떤 기능을 쓰나요?",
        "ground_truth": "Thickness(두께)는 Pad시킨 형상의 두께를 변경할 때 사용하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Part Design Transformation Feature의 Translation 기능은 무엇인가요?",
        "question": "만들어둔 형상을 특정 방향으로 정해진 거리만큼 옮기려면 어떤 기능인가요?",
        "ground_truth": "Translation(위치이동)은 기존 형상을 방향과 거리를 주어 위치를 변경할 때 사용하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Sketcher에서 동심원 구속 조건(Concentricity)이란 무엇인가요?",
        "question": "두 원의 가운데 점을 서로 똑같은 위치로 맞추고 싶을 때 거는 조건은 무엇인가요?",
        "ground_truth": "Concentricity는 두 선분 또는 원이 중심을 공유하고 그 중심에서의 거리가 일정하다는 것을 의미하며, Sketcher 도구바에서 'Concentricity' 옵션을 선택하여 적용합니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Assembly Design의 Fix Component는 어떤 기능인가요?",
        "question": "조립할 때 기준이 되는 부품 하나를 그 자리에서 못 움직이게 붙박이로 두려면 어떻게 하나요?",
        "ground_truth": "Fix Component(고정)는 파트의 위치를 현 위치에 고정하는 기능이며, 주로 기준점이 되는 Part를 고정할 때 사용합니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Assembly Design의 Offset Constraint는 언제 사용하나요?",
        "question": "두 부품을 딱 붙이지 않고 사이에 일정한 간격을 띄운 채로 묶어두고 싶습니다. 어떤 조건인가요?",
        "ground_truth": "Offset Constraint(거리구속)는 Part가 서로 붙어있지 않고 일정한 거리를 유지하며 구속될 경우 사용하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Assembly Design에서 Explode 기능은 무엇을 하나요?",
        "question": "조립된 부품들을 잠깐 사방으로 흩어놓아서 내부 구조를 보고 싶을 때 쓰는 것은?",
        "ground_truth": "Explode는 Part들을 공간으로 퍼트려주어 일시적으로 구속 조건을 해제시키는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Sketcher의 Snap to point(점에 맞추기) 기능은 무엇인가요?",
        "question": "스케치할 때 커서가 화면 눈금의 교차 지점에 착 달라붙게 하는 설정은 무엇인가요?",
        "ground_truth": "Snap to point(점에 맞추기)는 Sketcher 작업 시 격자의 교차점에 커서 포인트를 일치시켜주는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Sketcher 편집 도구 중 Break(끊기)는 어떤 기능인가요?",
        "question": "선 하나를 특정 지점에서 두 조각으로 나누고 싶은데, 그 지점 자체는 선에 포함되지 않게 하려면?",
        "ground_truth": "Break(끊기)는 직선 위의 한 점을 사용하여 직선을 분할하는 기능이며, 분할에 사용된 점은 직선에 포함되지 않습니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Sketcher 편집 도구 중 Symmetry(대칭)는 어떤 기능인가요?",
        "question": "그려둔 도형을 기준선 반대편에 거울처럼 뒤집어 똑같이 만들려면 어떤 기능을 쓰나요?",
        "ground_truth": "Symmetry(대칭)는 직선이나 Construction Line 또는 축을 사용하여 기존의 Sketcher 엘리먼트를 대칭적으로 반복 생성하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Sketcher 편집 도구 중 Corners(코너)는 어떤 기능인가요?",
        "question": "두 직선이 만나는 뾰족한 부분을 두 선에 접하는 둥근 원호로 바꾸려면 무엇을 쓰나요?",
        "ground_truth": "Corners(코너)는 트리밍 작업을 사용하여 두 직선 사이에 두 곡선에 접하는 원호(둥근 코너)를 생성하는 기능입니다."
    },
    {
        "type": "Hard/Paraphrased QA",
        "original": "Generative Shape Design Workbench는 어떤 기능을 하나요?",
        "question": "곡선을 이용해 자유롭게 휘어진 3차원 겉면을 만들려면 어떤 작업 환경을 써야 하나요?",
        "ground_truth": "Generative Shape Design은 Profile과 Curve를 이용해 3차원 자유곡면(sculptured surface)을 생성하는 기능입니다."
    },
]
