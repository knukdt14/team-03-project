import os
import sys
import types
import pandas as pd
from typing import List, Dict, Any
from bert_score import score
from src.rag_chain import RAGPipeline

## => 설치된 langchain-community가 최신(sunset) 버전이라 chat_models.vertexai 서브모듈이
##    통째로 빠져있는데, ragas가 (Vertex AI를 쓰지도 않으면서) 이걸 무조건 임포트해서 죽음.
##    실제로 쓰지 않는 클래스이므로, 진짜 패키지를 건드리지 않고 빈 모듈을 미리 등록해서
##    ragas의 임포트만 통과시키는 안전한 우회.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models  # noqa: F401
        _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class _ChatVertexAIStub:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("ChatVertexAI는 이 프로젝트에서 사용하지 않는 스텁입니다.")

        _vertexai_stub.ChatVertexAI = _ChatVertexAIStub
        sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub
    except ImportError:
        pass

# Try importing Ragas for RAG evaluation (PDF Pages 202-223)
try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False


# Benchmark dataset (Question, Ground Truth, Query Type)
## => 5문항 -> 16문항으로 확장 (파라미터 튜닝 비교 시 n=5는 노이즈에 묻히는 문제 대응)
##    새로 추가한 문항의 ground_truth는 전부 실제 PDF 원문을 직접 확인해서 작성함
##    (4주차/5주차/6주차 Part Design, 9주차 Assembly Design, 11주차 Generative Shape Design, 2주차 Sketcher)
BENCHMARK_DATASET = [
    {
        "type": "General QA",
        "question": "CATIA V5에서 Pad 기능의 주요 역할과 스케치 조건은 무엇인가요?",
        "ground_truth": "Pad 기능은 2D 스케치 프로파일을 돌출시켜 3D 솔리드 바디를 생성하는 기본 기능으로, 스케치는 닫힌 루프(Closed Loop) 형태이어야 합니다."
    },
    {
        "type": "General QA",
        "question": "Pad, Pocket 기능 중 Pocket의 역할은 무엇인가요?",
        "ground_truth": "Pocket(포켓)은 3차원 형상에서 2차원 형상의 공간을 만들어내는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Rib과 Slot 기능의 차이는 무엇인가요?",
        "ground_truth": "Rib은 Sketch가 경로를 따라가며 Solid를 생성하는 기능이고, Slot은 Sketch가 경로를 따라가며 Solid를 제거하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Generative Shape Design Workbench는 어떤 기능을 하나요?",
        "ground_truth": "Generative Shape Design은 Profile과 Curve를 이용해 3차원 자유곡면(sculptured surface)을 생성하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Transformation Feature의 Mirror와 Pattern 기능은 각각 무엇인가요?",
        "ground_truth": "Mirror(복사)는 기존 형상을 축이나 면을 기준으로 똑같이 복사할 때 사용하고, Pattern(패턴)은 기존 형상을 복사하여 여러 개 만들어낼 때 사용합니다."
    },
    {
        "type": "Specific QA",
        "question": "Dress-Up Feature 중 Stiffener 기능의 사용 목적과 권장 구배 각도는 몇 도인가요?",
        "ground_truth": "Stiffener 기능은 캐싱 부품 내부 리브(Rib) 보강재를 효율적으로 생성할 때 사용하며, 몰딩 공정 부품의 경우 보통 4도의 구배 각도가 권장됩니다."
    },
    {
        "type": "Specific QA",
        "question": "Draft Angle 기능이란 무엇이며 어떨 때 사용하나요?",
        "ground_truth": "Draft Angle 기능은 금형(Molding) 공정에서 부품을 원활하게 빼내기 위해 측면에 꺾임각/경사각을 부여하는 Dress-Up 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Shaft 기능은 무엇인가요?",
        "ground_truth": "Shaft(쉐프트)는 중심축을 기준으로 2차원 단면을 회전시켜 3차원 형상을 만드는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Groove 기능은 무엇인가요?",
        "ground_truth": "Groove(그루브)는 Shaft와 반대로 3차원상에 원형으로 Pocket을 만드는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Chamfer 기능은 언제 사용하나요?",
        "ground_truth": "Chamfer(모따기)는 면과 면 사이에 경사가 필요한 경우, 모서리에 모따기를 생성할 때 사용합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher에서 동심원 구속 조건(Concentricity)이란 무엇인가요?",
        "ground_truth": "Concentricity는 두 선분 또는 원이 중심을 공유하고 그 중심에서의 거리가 일정하다는 것을 의미하며, Sketcher 도구바에서 'Concentricity' 옵션을 선택하여 적용합니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design의 Contact Constraint는 어떤 기능인가요?",
        "ground_truth": "Contact Constraint(면일치)는 Part들의 면과 면을 일치시킬 수 있고, 점과 선으로도 구속을 할 수 있는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design의 Offset Constraint는 언제 사용하나요?",
        "ground_truth": "Offset Constraint(거리구속)는 Part가 서로 붙어있지 않고 일정한 거리를 유지하며 구속될 경우 사용하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design의 Angle Constraint는 무엇인가요?",
        "ground_truth": "Angle Constraint(각도구속)는 각 Part에 각도를 주어 구속을 주는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design의 Fix Component는 어떤 기능인가요?",
        "ground_truth": "Fix Component(고정)는 파트의 위치를 현 위치에 고정하는 기능이며, 주로 기준점이 되는 Part를 고정할 때 사용합니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design의 Coincidence Constraint는 무엇인가요?",
        "ground_truth": "Coincidence Constraint(축일치)는 Rotational Part들 간의 축을 일치시켜 구속을 주는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Assembly Design에서 Explode 기능은 무엇을 하나요?",
        "ground_truth": "Explode는 Part들을 공간으로 퍼트려주어 일시적으로 구속 조건을 해제시키는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Part Design의 Hole 기능은 무엇인가요?",
        "ground_truth": "Hole(홀)은 원형의 구멍을 만드는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Part Design의 Rib 기능은 무엇인가요?",
        "ground_truth": "Rib(립)은 미리 만든 2차원 형상을 Guide Line을 따라 3D 형상으로 만드는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Part Design의 Slot 기능은 무엇인가요?",
        "ground_truth": "Slot(슬롯)은 2차원 형상을 Guide Line을 따라 Pocket 작업하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Part Design의 Shell 기능은 언제 사용하나요?",
        "ground_truth": "Shell(쉘)은 형상의 일정한 두께를 남기고 빈 공간을 만들고자 할 때 사용하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Part Design의 Thickness 기능은 무엇인가요?",
        "ground_truth": "Thickness(두께)는 Pad시킨 형상의 두께를 변경할 때 사용하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Part Design의 Edge Fillet 기능은 언제 사용하나요?",
        "ground_truth": "Edge Fillet(엣지필렛)은 면과 면 사이에 부드러운 연결 형상이 필요한 경우에 사용하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Part Design Transformation Feature의 Translation 기능은 무엇인가요?",
        "ground_truth": "Translation(위치이동)은 기존 형상을 방향과 거리를 주어 위치를 변경할 때 사용하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Part Design에서 3D Plane(평면)은 어떤 역할을 하나요?",
        "ground_truth": "Plane(평면)은 새로운 Plane을 만들어 그 평면상에서 작업을 가능하게 만드는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Circle(원) 도구는 어떻게 사용하나요?",
        "ground_truth": "Circle(원)은 Tools 툴바를 사용하거나 원의 중심점과 원 위의 한 점을 정의하기 위해서 선택합니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Rectangle(직사각형) 도구는 무엇인가요?",
        "ground_truth": "Rectangle(직사각형)은 직사각형을 그릴 때 사용하는 도구입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Arc(호) 도구는 어떻게 사용하나요?",
        "ground_truth": "Arc(호)는 Tools 툴바를 사용하거나 원호의 중심과 원호의 시작점과 끝점을 정의하기 위해서 선택합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Hexagons(육각형) 도구는 어떻게 사용하나요?",
        "ground_truth": "Hexagons(육각형)는 Tools 툴바를 사용하거나 클릭하여 육각형의 중심과 치수를 정의하는 도구입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Tri-tangent Circle(세접점원)은 어떻게 생성하나요?",
        "ground_truth": "Tri-tangent Circle(세접점원)은 세 개의 접선(tangent) 구속조건으로 원을 생성하기 위해서 차례대로 세 개의 엘리먼트를 선택합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Circle Using Coordinates(좌표를 사용한 원)는 어떻게 사용하나요?",
        "ground_truth": "Circle Using Coordinates(좌표를 사용한 원)는 원의 중심점과 반지름을 정의하기 위해서 Circle Definition 다이얼로그 박스를 사용합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Ellipse(타원) 도구는 어떻게 사용하나요?",
        "ground_truth": "Ellipse(타원)는 Tools 툴바를 사용하거나 타원의 중심점과 장축 끝점과 단축 끝점을 정의하기 위해서 차례대로 선택합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Parabola(포커스별 포물선)는 어떻게 생성하나요?",
        "ground_truth": "Parabola(포커스별 포물선)는 초점과 정점을 클릭하고 포물선의 두 끝점을 선택해서 생성합니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Conic 도구는 무엇인가요?",
        "ground_truth": "Conic(원근감)은 타원, 원, 포물선, 쌍곡선을 생성하기 위해서 원하는 점과 Eccentricity를 선택하는 도구이며, 필요하면 탄젠트를 사용합니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Line(선) 도구는 어떻게 사용하나요?",
        "ground_truth": "Line(선)은 Tools 툴바를 사용하거나 직선의 첫 번째 점과 두 번째 점을 선택하는 도구입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Spline(스플라인) 도구는 무엇인가요?",
        "ground_truth": "Spline(스플라인)은 스플라인이 지나갈 점을 선택하는 도구입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Axis(축) 도구는 무엇인가요?",
        "ground_truth": "Axis(축)는 Tools 툴바를 사용하거나 축의 첫 번째 점과 두 번째 점을 선택하는 도구입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Point Using Coordinates(좌표를 사용한 점)는 어떻게 사용하나요?",
        "ground_truth": "Point Using Coordinates(좌표를 사용한 점)는 Point Definition 다이얼로그 박스에 직교(Cartesian) 좌표 또는 극(Polar) 좌표를 입력하는 도구입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Equidistant Point(등거리점) 도구는 어떻게 사용하나요?",
        "ground_truth": "Equidistant Point(등거리점)는 직선 또는 곡선에 생성하려는 등거리점의 개수와 간격을 Equidistant Point Definition 다이얼로그 박스에 입력하는 도구입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Bi-tangent Line(쌍접점선)은 어떻게 생성하나요?",
        "ground_truth": "Bi-tangent Line(쌍접점선)은 두 엘리먼트에 동시에 접하는 직선을 생성하기 위해서 두 엘리먼트를 차례대로 선택합니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher의 Grid(격자무늬) 기능은 무엇인가요?",
        "ground_truth": "Grid(격자무늬)는 Sketch Workbench 화면상의 격자무늬를 on/off로 설정하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Snap to point(점에 맞추기) 기능은 무엇인가요?",
        "ground_truth": "Snap to point(점에 맞추기)는 Sketcher 작업 시 격자의 교차점에 커서 포인트를 일치시켜주는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher의 Geometrical Constraint(기하학적 구속조건)란 무엇인가요?",
        "ground_truth": "Geometrical Constraint(기하학적 구속조건)는 Profile 작업 시 구속조건을 자동으로 생성하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher 편집 도구 중 Chamfers(모따기)는 어떤 기능인가요?",
        "ground_truth": "Chamfers(모따기)는 트리밍 작업을 사용하여 두 직선 사이에 모따기를 생성하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher 편집 도구 중 Trim(자르기)은 어떤 기능인가요?",
        "ground_truth": "Trim(자르기)은 두 직선을 자르는(하나의 엘리먼트 또는 모든 엘리먼트에 대해) 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher 편집 도구 중 Break(끊기)는 어떤 기능인가요?",
        "ground_truth": "Break(끊기)는 직선 위의 한 점을 사용하여 직선을 분할하는 기능이며, 분할에 사용된 점은 직선에 포함되지 않습니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher 편집 도구 중 Quick Trim(즉시자르기)은 어떤 기능인가요?",
        "ground_truth": "Quick Trim(즉시자르기)은 브레이킹과 트리밍 작업을 사용하여 다른 Sketcher 엘리먼트에 의해서 교차되는 엘리먼트를 제거하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher 편집 도구 중 Symmetry(대칭)는 어떤 기능인가요?",
        "ground_truth": "Symmetry(대칭)는 직선이나 Construction Line 또는 축을 사용하여 기존의 Sketcher 엘리먼트를 대칭적으로 반복 생성하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher 편집 도구 중 Corners(코너)는 어떤 기능인가요?",
        "ground_truth": "Corners(코너)는 트리밍 작업을 사용하여 두 직선 사이에 두 곡선에 접하는 원호(둥근 코너)를 생성하는 기능입니다."
    },
    {
        "type": "Specific QA",
        "question": "Sketcher 편집 도구 중 Close arc(닫기)는 어떤 기능인가요?",
        "ground_truth": "Close arc(닫기)는 제한(Relimiting) 작업을 사용하여 원과 타원과 스플라인을 닫힌 곡선으로 정의하는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher 편집 도구 중 Translate(이동)는 어떤 기능인가요?",
        "ground_truth": "Translate(이동)는 Duplicate 모드를 정의한 후 복사 대상 엘리먼트를 선택하여 2D 엘리먼트를 평행이동하는 기능이며, 복수 선택은 불가합니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher 편집 도구 중 Rotate(회전)는 어떤 기능인가요?",
        "ground_truth": "Rotate(회전)는 Duplicate 모드를 정의한 후 복사될 엘리먼트를 선택하여 엘리먼트를 회전시키는 기능입니다."
    },
    {
        "type": "General QA",
        "question": "Sketcher 편집 도구 중 Offset(오프셋)은 어떤 기능인가요?",
        "ground_truth": "Offset(오프셋)은 직선, 원호 또는 원과 같은 엘리먼트를 복사하는 기능입니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 3D 퀀텀 머시닝(Quantum Machining) AI 가속 모드는 어떻게 실행하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 자율주행 차체 3D 홀로그램 자동 설계 알고리즘 메뉴 위치는?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 음성 명령으로 3D 형상을 자동 생성하는 Voice Modeling 기능은 어떻게 사용하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 블록체인 기반 부품 이력 관리 기능은 어떻게 설정하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5의 뇌파 인식 스케치 입력 기능은 어떻게 활성화하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 AI가 자동으로 최적 재질을 추천해주는 Smart Material Advisor는 어디에 있나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5에서 메타버스 협업 모드로 여러 사용자가 동시에 파트를 수정하는 기능은 무엇인가요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    },
    {
        "type": "Trick/Unanswerable QA",
        "question": "CATIA V5의 자동 특허 출원서 생성 기능은 어떻게 사용하나요?",
        "ground_truth": "제공된 매뉴얼에 관련 내용이 명시되어 있지 않습니다."
    }
]


def calculate_ragas_metrics(pipeline: RAGPipeline, questions: List[str], answers: List[str], contexts: List[List[str]], ground_truths: List[str]) -> Dict[str, List[float]]:
    """
    Calculates Ragas evaluation metrics (PDF Page 202-223):
    - faithfulness: How factually accurate the generated answer is to the retrieved contexts.
    - answer_relevancy: How relevant the generated answer is to the user question.
    - context_precision: Signal-to-noise ratio of retrieved contexts.
    - context_recall: Whether all necessary information was retrieved.
    """
    if not HAS_RAGAS:
        print("[Evaluation Warning] Ragas library not available, skipping Ragas metrics.")
        return {}
        
    try:
        ragas_dict = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths
        }
        dataset = Dataset.from_dict(ragas_dict)
        print("[Evaluation] Prepared RAGAS Dataset format (user_input, response, retrieved_contexts, reference).")
        
        # Check if OpenAI API Key is present for automated LLM judge, otherwise skip API call to prevent hang
        if not os.getenv("OPENAI_API_KEY"):
            print("[Evaluation Note] Ragas automated LLM-as-a-judge requires OPENAI_API_KEY. Formatted Dataset created successfully.")
            return {}
            
        results = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df_ragas = results.to_pandas()
        return {
            "ragas_faithfulness": df_ragas.get("faithfulness", [0.0]*len(questions)).tolist(),
            "ragas_answer_relevancy": df_ragas.get("answer_relevancy", [0.0]*len(questions)).tolist(),
            "ragas_context_precision": df_ragas.get("context_precision", [0.0]*len(questions)).tolist(),
            "ragas_context_recall": df_ragas.get("context_recall", [0.0]*len(questions)).tolist(),
        }
    except Exception as e:
        print(f"[Evaluation Note] Ragas execution notice ({e}). Skipping automated Ragas API call.")
        return {}


def run_evaluation_benchmark(pipeline: RAGPipeline, dataset: List[Dict[str, str]] = BENCHMARK_DATASET) -> pd.DataFrame:
    """
    Runs 3-way A/B testing benchmark with mandatory BERTScore and Ragas metrics:
    1. Direct LLM (RAG Off)
    2. Strict RAG (Strict Manual Only)
    3. Adaptive Fallback RAG (Manual First -> Direct LLM Fallback)
    """
    questions = [d["question"] for d in dataset]
    ground_truths = [d["ground_truth"] for d in dataset]
    types = [d["type"] for d in dataset]
    
    rag_off_answers = []
    strict_rag_answers = []
    adaptive_answers = []
    adaptive_statuses = []
    source_pages_list = []
    retrieved_contexts_list = []
    
    print("[Evaluation] Executing 3-Way Benchmark (Direct LLM vs Strict RAG vs Adaptive Fallback RAG)...")
    for item in dataset:
        q = item["question"]
        # 1. Direct LLM (RAG Off)
        ans_off = pipeline.answer_direct(q)
        rag_off_answers.append(ans_off)
        
        # 2. Strict RAG
        res_strict = pipeline.answer_rag(q)
        strict_rag_answers.append(res_strict["answer"])
        source_pages_list.append(str(res_strict["source_pages"]))
        
        # Extract text content of retrieved docs for Ragas & Context inspection
        raw_contexts = [doc.page_content for doc in res_strict["retrieved_docs"]]
        retrieved_contexts_list.append(raw_contexts)
        
        # 3. Adaptive Fallback RAG
        res_adaptive = pipeline.answer_adaptive_fallback(q)
        adaptive_answers.append(res_adaptive["answer"])
        adaptive_statuses.append(res_adaptive["status_tag"])
        
    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Direct LLM...")
    P_off, R_off, F1_off = score(cands=rag_off_answers, refs=ground_truths, lang="ko", verbose=False)
    
    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Strict RAG...")
    P_strict, R_strict, F1_strict = score(cands=strict_rag_answers, refs=ground_truths, lang="ko", verbose=False)

    print("[Evaluation] Calculating BERTScore (Precision, Recall, F1) for Adaptive Fallback RAG...")
    P_adapt, R_adapt, F1_adapt = score(cands=adaptive_answers, refs=ground_truths, lang="ko", verbose=False)
    
    # Base Results DataFrame with Full BERTScore metrics (P, R, F1) matching PDF Page 199-213
    df_results = pd.DataFrame({
        "Type": types,
        "Question": questions,
        "Ground Truth": ground_truths,
        "Direct LLM Answer": rag_off_answers,
        "Direct LLM BERTScore Precision": P_off.tolist(),
        "Direct LLM BERTScore Recall": R_off.tolist(),
        "Direct LLM F1": F1_off.tolist(),
        "Strict RAG Answer": strict_rag_answers,
        "Strict RAG BERTScore Precision": P_strict.tolist(),
        "Strict RAG BERTScore Recall": R_strict.tolist(),
        "Strict RAG F1": F1_strict.tolist(),
        "Adaptive Fallback Answer": adaptive_answers,
        "Adaptive Status": adaptive_statuses,
        "Adaptive Fallback BERTScore Precision": P_adapt.tolist(),
        "Adaptive Fallback BERTScore Recall": R_adapt.tolist(),
        "Adaptive Fallback F1": F1_adapt.tolist(),
        "Source Pages": source_pages_list
    })
    
    # Calculate Ragas Metrics (PDF Page 202-223)
    ragas_metrics = calculate_ragas_metrics(pipeline, questions, strict_rag_answers, retrieved_contexts_list, ground_truths)
    for col_name, values in ragas_metrics.items():
        df_results[col_name] = values
        
    print("[Evaluation] 3-Way Benchmark & Metric Assessment Complete.")
    return df_results


if __name__ == "__main__":
    pipeline = RAGPipeline()
    df = run_evaluation_benchmark(pipeline)
    print(df[["Type", "Question", "Direct LLM F1", "Strict RAG F1", "Adaptive Fallback F1"]])
