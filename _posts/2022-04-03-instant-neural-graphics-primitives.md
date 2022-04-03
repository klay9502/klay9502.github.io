---
layout: post
title: NVIDIA Instant NeRF 환경을 구축해보자
image: post_ngp.jpg
data: 2021-05-13 19:37:00 +0900
tags: [AI, Study, NVIDIA, Github, Algorithm]
categories: study
---

##### 시작하기 앞서

거의 1년 만에 포스팅한다. 작년부터 쭉 Github를 자주 활용하고자 노력했지만 정작 본격적으로 사용한 건 올해부터인 것 같다. 앞으로는 포스팅도 자주 하고 git push도 자주 하게끔 좀 더 분발해야겠다. 이번 포스트는 영상애니메이션과 교수님의 부탁으로 작업한 것을 정리한 것이다. 다음에 다시 작업할 일이 생기면 좀 더 수월하게 할 수 있도록, 그리고 비전공자들도 더욱 쉽게 환경을 구축할 수 있도록 이 포스트에 정리한다.

> 본 포스터는 NVIDIA Github에 있는 해당 리포지토리를 보기 쉽게 정리한 것이다. 전공자들은 본 포스터보고 구축하는 것보다 해당 README.md를 보고 구축하는 것을 권장한다.<br>
> 링크 - [Github-NVlabs/instant-ngp][link_1]

***

##### Instant NeRF(Neural Radiance Fields)란 무엇인가?

Instant NeRF란 NVIDIA사에서 개발한 기술로, 여러 개의 2D 이미지를 신경 네트워크를 사용해 하나의 3D 모델로 변환하는 기술이다. 2D 이미지를 3D 모델로 변환하는 기술은 그렇게 새로운 기술은 아니다. 하지만 이번에 개발된 Instant NeRF는 렌더링 시간을 매우 빠른 속도로 단축함과 동시에 높은 퀄리티의 결과물을 얻을 수 있다. 또한 NVIDIA CUDA Toolkit을 사용하여 빠른 속도로 훈련 및 실행할 수 있다. 본 기술에 대해 좀 더 자세하게 알고 싶다면 NVIDIA Blog를 참고하자.<br><br>

> NVIDIA BLOG - [링크][link_2]

***
# 환경 구축

본 포스터에는 기초적인 요구 사항들도 간단히 기재할 계획이다. 기술이 기술인 만큼 비전공자들도 사용할 수 있기 때문이다.

*** 

##### 요구 사항 및 테스트 환경

먼저 요구 사항 및 테스트 환경이다. 본 포스터에 기재된 요구 사항들은 NVIDIA사에서 요구하는 사항들이다.

* <strong><u>Windows</u></strong>나 <strong><u>Linux</u></strong> 운영체제 - 본 포스터는 Windows를 기반으로 한다.
* <strong><u>NVIDIA GPU</u></strong> - 텐서 코어를 사용할 수 있는 GPU라면 성능을 향상할 수 있다. 본 테스트에는 RTX 3090을 사용하였다.
* <strong><u>C++14가 가능한 컴파일러</u></strong> - Windows에서 사용할 것이므로 NVIDIA에서 권장하는 Visual Studio 2019를 사용하였다.
* <strong><u>CUDA v10.2 이상</u></strong> 및 <strong><u>CMake v3.21 이상</u></strong>
* <strong><u>Python 3.7 이상</u></strong> - 파이썬 설치에 관해서는 해당 문서가 워낙 많으니 생략하겠다.

***

##### 요구 사항 충족하기

요구 사항에서 C++14가 가능한 컴파일러를 요구했기 때문에 「Visual Studio 2019」를 설치할 것이다. 본래 상위버전(2022)을 사용했지만 혹시 모를 버그를 만연에 방지하기 위해 2019를 사용하였다. [링크](https://docs.microsoft.com/ko-kr/visualstudio/releases/2019/release-notes)를 통해 설치하면 된다.

CUDA Toolkit는 [링크](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)를 통해 다운받으면 된다. 사이트에서 Version은 CUDA 버전이 아닌 운영체제 버전을 묻는 것이니 본인 환경의 운영체제에 맞게 설치해주면 된다. 본 테스트에는 Windows 10을 사용했으므로 10을 선택한다.

CMake 다운로드 [링크](https://cmake.org/download/)이다. 소스가 아닌 바이너리로 다운 받으면 된다. 대부분 64비트를 사용할 것이므로 64비트 인스톨러를 다운받으면 된다.

***

##### 컴파일하기

명령 프롬프트(cmd)를 실행하여 아래 명령어를 입력하여 해당 리포지토리를 복제하자. 복제하기 위해서는 git이 설치되어 있어야한다. 설치가 안 되어있다면 [링크](https://git-scm.com/downloads)에서 다운 후 설치하고 진행하자. 원하는 위치에 복제하면 되지만 본 포스터에는 C드라이브에 ngp 폴더를 만들어서 하겠다.

```sh
cd C:\ngp
git clone --recursive https://github.com/nvlabs/instant-ngp
cd instant-ngp
```

그런 뒤 CMake를 이용해 프로젝트를 빌드하자

```sh
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

아무 문제없이 빌드가 되었다면 정상적으로 컴파일 된 것이다!

***

##### 랜더링해보기

복제한 리포지토리 안에는 테스트 씬이 포함되어 있다. 시험삼아 「NeRF fox」를 랜더링 해볼 것이다.

```sh
.\build\testbed.exe --scene data\nerf\fox
```

<br>
![]({{site.baseurl}}/images/fox.png)
<br>

위 이미지처럼 나온다면 정상적으로 랜더링이 된 것이다. [1]

***

##### 마치며
본 기술을 교수님을 통해 처음 알게 되었을 때 무척 신기했다. AI를 전공하고 있지만 이런 기술을 새로이 접할 때마다 기술의 발전이 경이로울 정도이다. 이제는 정말로 AI가 기존 인간들의 직업들을 대체할 수 있겠다고 조심스레 생각해본다.

사실 전공자들은 README.md 만으로도 손쉽게 구축할 수 있을 것이다. 하지만 기술이 발전하면서 AI가 컴퓨터공학만의 소유물이 아니게 되면서 비전공자들도 AI를 접할 일이 많아졌다. 그런 비전공자들도 더욱 쉽게 AI를 접할 수 있도록 본 포스트를 작성해본다. 많은 도움은 안 되겠지만 본 포스터가 조금이나마 도움이 되었으면 좋겠다.

***

##### 출처

* [대문이미지] - NVIDIA
* [1] - [NVIDIA/NVlabs/instant-ngp][link_3]

[link_1]: https://github.com/NVlabs/instant-ngp
[link_2]: https://blogs.nvidia.com/blog/2022/03/25/instant-nerf-research-3d-ai/
[link_3]: https://github.com/NVlabs/instant-ngp/blob/master/docs/assets_readme/fox.png