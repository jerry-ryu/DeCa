# DeCa
## 1. Project Description
Depth estimation and computer vision based autonomous driving car (Depth + Car)

## 2. Baseline
https://github.com/shariqfarooq123/AdaBins

## 3. convention
### Commit Message

```
type : subject       -> 필수

body                 -> 선택
```

- type 은 아래 중 하나 선택하여 기재
    - init: repository 생성
    - feat : 새로운 기능 추가
    - fix : 버그 수정, 기능 수정
    - refactor : 리팩토링 (변수명 수정, 파일 구조 수정등)
    - test : 테스트 코드, 리팩토링 테스트 코드 추가
    - docs : 훈련 및 테스트에 필요한 문서 수정
    - chore : git 관리 문서 수정, 패키지 매니저 수정 (gitignore 수정 등)
- Subject 는 50자를 넘기지 않고, 대문자로 시작, 명령어로 작성.
- 예시
    
    ```
    
    Chore : Moidfy .gitignore          # .gitignore 수정
    
    Docs : Update result             # README.md의 Result 업데이트
    
    feat : Add CustomAugmentation at dataset.py  # dataset.py에 CustomAugmentation 추가
    ```
    
- Body는 선택사항, 부연설명이 필요하거나 커밋의 이유를 설명할 때
    - 제목과 구분되기 위해 한칸을 띄워 작성합니다.
    - 각 줄은 72자를 넘기지 않습니다.
    - 본문은 꼭 영어로 작성할 필요는 없습니다.
 
### Branch
실험별로 branch 생성 및 결과 정리
- Master → 제출할 프로젝트 / 모든 모델을 실행시킬 수 있는 branch
- hotfix → Master에서 가져와서  버그 or 하이퍼파리미터 등 소소한 부분 수정
- Develop →  새로운 구조를 실험해볼 때 파는 branch
- Feature → Develop 브랜치에서 추가 실험이 필요할 때 파는 branch (Develop의 가지)

### Issue / PR
**Issue**:
GitHub Issues는 프로젝트를 더 작은 작업으로 나누어 계획할 수 있게 해줍니다. 각 작업은 팀원에게 할당하여 책임을 명확히 할 수 있습니다. 또한, 각 작업의 진행 상황을 추적하고 의견을 나누고 업데이트할 수 있기 때문에 프로젝트의 진행을 효율적으로 관리할 수 있습니다.

**PR**: 
GitHub Pull Request는 내가 작업한 코드를 기존 코드베이스에 병합하기 전에 팀원과 함께 코드를 검토하는 과정입니다. 피드백을 통해 코드의 품질을 유지하고, 팀원과 의견을 공유하며, 더 나은 코드를 작성할 수 있습니다.
