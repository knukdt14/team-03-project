# CATIA 핵심 사실 보강 문서

이 문서는 제공된 CATIA 교육 PDF의 검색 실패 사례를 보강하기 위해, 원문에 직접 확인된 핵심 사실만 정리한 문서입니다.

## 파일 확장자
CATIA Workbench 파일 확장자는 다음과 같습니다.
- Part Design: .CATPart
- Drawing: .CATDrawing
- Assembly Design: .CATProduct
출처: 2주차 (Sketcher).pdf, 6페이지.

## Sketcher 과구속
스케치를 구속하는 중 보라색으로 변하면 over-constrained(과구속) 상태입니다.
출처: CATIA V5 Lectures.pdf, 20페이지.

## 추가 Body와 Boolean 연산
추가 Body는 Insert -> New Body 기능으로 생성합니다.
Boolean operations(join, subtract, intersect)은 main PartBody와 같은 Part 안의 다른 Body 사이에서만 적용할 수 있습니다.
출처: CATIA V5 Lectures.pdf, 32페이지.

## Generative Part Structural Analysis
Generative Part Structural Analysis는 mesh control이 매우 제한적이며 solid geometry에만 적용할 수 있습니다.
출처: CATIA V5 Lectures.pdf, 54페이지.

## STEP 형식
CATPart 또는 CATProduct 문서의 데이터는 STEP AP203 또는 AP214 형식으로 저장할 수 있습니다.
출처: basug_Knowledgeware.pdf, 298페이지.

## Constraints
Constraints Toolbar에는 Assembly constraints 기능이 있으며, Coincidence와 Contact constraint를 사용할 수 있습니다.
출처: Assembly Design.pdf, 3페이지 및 14페이지.
