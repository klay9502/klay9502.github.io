---
layout: post
title: 멀티 암드 밴딧(Multi Armed Bandit)을 배워보자
image: post_MAB.jpg
data: 2021-05-13 19:37:00 +0900
tags: [AI, Study, MAB, Reinforcement learning, UCB, Algorithm]
cartegories: study
---

##### 시작하기 앞서

연구회에서 본격적으로 무언가를 배우기 시작한 알고리즘이다. Multi Armed Bandit 알고리즘(이하 MAB 알고리즘)은 Q-Learning 알고리즘과 함께 강화학습의 기초 알고리즘 중 하나이며, 하나씩 차근차근 배워가는 나에게 공부하기 안성맞춤인 알고리즘이다. 사실 MAB 알고리즘을 공부한지는 한달이 조금 넘어가지만 대학 중간고사와 함께 이런저런 이유로 포스팅이 조금 늦어졌다. 복습도 할겸 이 포스트에 깔끔하게 정리할 생각이다.

***

##### 멀티 암드 밴딧(Multi Armed Bandit)이 무엇인가?

MAB 알고리즘은 카지노의 슬롯머신을 시작으로 만들어진 알고리즘이다. 슬롯머신 마다 당첨될 확률이 다 다르지만 우리는 어느 슬롯머신이 확률이 높은지 알 수가 없다. 만약 고객들이 알고 있다면 그 카지노는 홀딱 망할 것이다. 하지만 꾸준하게 카지노에서 슬롯머신의 팔(Arm)을 당긴다면 경험적으로 어느 슬롯머신이 더 확률이 좋은지 알 수 있다. MAB 알고리즘은 그런 경험을 토대로 <strong><u>「어떻게하면 효율적으로 슬롯머신을 당겨 높은 수익을 낼 수 있는가?」 라는 의문에서 고안된 알고리즘</u></strong> 되시겠다.
<br><br>
MAB 알고리즘은 강화학습의 핵심인 <strong><u>「탐색(Exploration)」</u></strong>과 <strong><u>「활용(Exploitation)」</u></strong>을 모두 다루고 있다. 거기에 다른 강화학습 알고리즘에 비해 <strong><u>무척 쉽다!</u></strong> 그리고 카지노에서 시작된 알고리즘인 만큼 경제적인 면에서 무척 유용하다. <strong><u>한마디로 돈이된다!</u></strong> 강화학습을 공부하는 입장에서는 아주 유익한 알고리즘이니 꼭 공부하도록 하자.

***

# 3가지의 전략
우리는 총 3가지의 전략을 취함으로써 그 결과를 바탕으로 공부할 것이다. 그 3가지 전략은 다음과 같다.

* <strong><u>Greedy (탐욕 알고리즘)</u></strong>
* <strong><u>𝜀-Greedy (입실론 그리디 알고리즘)</u></strong>
* <strong><u>UCB (Upper Confidence Bound)</u></strong>

전략을 취하기전에 아래 사항을 전재로 한다.

> 1. 슬롯머신을 당기는 머신을 <strong>'A'</strong>라고 칭하고 'A'는 위의 3가지 전략을 토대로 동작한다.<br>
> 2. 존재하는 슬롯머신은 총 <strong>5가지</strong>이다.<br>
> 3. 각 슬롯머신의 확률은 'A'가 <strong>알지 못한다.</strong><br>
> 4. 각 슬롯머신의 확률은 <strong>표준편차에 따른 정규분포</strong>를 가진다.<br>
> 5. 'A'가 슬롯머신을 당길 수 있는 횟수의 최대값은 <strong>5000회</strong>이다.

자, 이제 차례대로 한번 공부해보자.

***

##### 전략 1. Greedy (탐욕 알고리즘)
Greedy 알고리즘은 <strong><u>각 슬롯을 한번씩 당겨본 후, 결과가 좋은 슬롯에 올인</u></strong>하는 전략이다. 다음과 같이 5가지의 슬롯머신이 결과값이 나왔다고 가정해보자.

* A. 0.1412
* B. 0.5423
* C. 0.2134
* D. 0.7864
* E. 0.0013

'A'는 Greedy 알고리즘에 의해 남은 기회 4995회를 D 슬롯머신에 올인할 것이다. Greedy 알고리즘을 수식으로 나타내면 다음과 같다.

<br>
\\[
\Large
q_{*}(a)\approx E[R_{t}|A_{t}=a]\cdot
\\]
<br>
\\[
\Large
A_{t}\approx argmax_{a}Q_{t}(a)
\\]
<br>

$ a $ 는 우리가 고른 슬롯머신을, $ t $ 는 시점을 뜻하는 변수이다.<br> 
즉, $ E[R_{t}|A_{t}=a]\cdot $ 는 $ t $ 시점의 $ A $ (선택할 슬롯머신, Action) 이 $ a $ 라는 슬롯머신 일 때의 $ R $ (Reward, 보상) 이라는 뜻이다.<br>
첫번째 수식을 요약을 하자면 「$ q_{*}(a) $ 은 $ a $ 를 선택했을 때, $ t $ 시점에서 얻을 수 있는 기대보상이다.」 라고 말할 수 있다.

두번째 수식은 $ Q_{t}(a) $ 의 최대값을 취하는 $ A_{t} $ (Action)을 취해라는 말이다. ($ argmax_{a} $ 는 최대값을 뜻함.) $ Q_{t}(a) $ 의 수식은 다음과 같다.

<br>
\\[
\Large
Q_{t}(a)\approx \frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}}
\\]
<br>
\\[
\Large
=\frac{\sum_{i=1}^{t-1}R_{i}\cdot I}{\sum_{i=1}^{t-1} I}
\\]
<br>
\\[
\Large
I=\mathfrak{1}_{A}:i=a\rightarrow 0
\\]
<br>

$ I $ 는 Indicator Function 이다. (AI가 a라는 식이 성립할 1의 값을, 그렇지 않으면 0을 반환하는 함수)<br>
두번째 식을 요약하면 $ t $ 전 까지 $ a $ 가 선택되었을 때의 보상의 총합을 $ t $ 전 까지 $ a $ 가 선택된 횟수의 총합을 나눈 것이 $ Q_{t}(a) $ , 즉 $ q_{*}(a) $ 의 추정값을 말한다.<br>

자, 수식으로 표현하니 엄청 복잡해졌다. 하지만 우리는 개념만 이해하면 된다. 왜냐하면 Greedy 전략은 문제가 심각한 전략이기 때문이다. 과연 한 번씩만 테스트를 하고 그 결과가 최선이라고 말할 수 있을까? 운이 나빠서 그 테스트결과가 엉망일 때는 어떻게 할 것인가? 더 좋은 슬롯머신이 있는데 테스트를 바탕으로 결과가 좋은 기존의 슬롯머신을 계속 고집할 것인가? 이건 최선의 결과를 가져왔다고 말할 수 없다. Greedy 전략은 강화학습의 핵심 중 「활용(Exploitation)」을 다루고 있지만 <strong><u>「탐색(Exploration)」</u></strong> 은 충분히 다루지 않는다. 따라서 가장 최적화된 전략이라고 말할 수 없을 것이다.

***

##### 전략 2. 𝜀-Greedy (입실론 그리디 알고리즘)

𝜀-Greedy 전략은 앞에서 언급한 Greedy 전략의 치명적인 단점인 「탐색(Exploration)」의 부재를 보완한 전략이다. 𝜀-Greedy 전략은 <strong><u>일정 확률로 랜덤하게 기존의 슬롯머신이 아닌 다른 슬롯머신을 선택</u></strong>하게 한다. 'A'는 일종의 모험을 하는 샘이다. 𝜀(입실론)이라는 하이퍼파라미터가 이 확률을 뜻한다. (예로 50%의 확률로 탐색을 한다면 'A'의 𝜀은 0.5가 되고 50% 확률로 탐색을 할 것이다.)

<br>
\\[
\Large
A \leftarrow\begin{cases}
& argmax_{a}Q_{t}(a) \text{ with probability 1-𝜀 }
& \text{any action (a)} \text{ with probability 𝜀 }
\end{cases}
\\]
<br>

𝜀(입실론) 값을 조절해서 얼마나 자주 탐색을 할 것인가 조절할 수 있다. (값이 클수록 높은 확률로 탐색을 함)

<br>
![]({{site.baseurl}}/images/e-greedy.png)
<br>

위 이미지를 참고하면 적당히 탐색을 함으로써 탐색을 하지 않는 Greedy 전략보다 더나은 결과를 가져오는 것을 확인할 수 있다. [1]

앞서 언급한 Greedy 전략보다는 𝜀-Greedy 전략이 효율적이고 위험성이 적다. 이대로 𝜀-Greedy 전략을 사용해도 괜찮을 것 같다. 하지만 우리는 좀 더 나은 방법을 찾기 위해 공부하지 않는가? 조금만 더 생각해보자.

***

##### 전략 3. UCB (Upper Confidence Bound)

𝜀-Greedy 전략의 문제점에 대해 생각해보자. 꼭 확률적으로 탐색을 해야할까? 여러번 탐색을 했는데 그 결과가 형편없어도 계속 꾸준히 탐색을 해야할까? 이건 너무 비효율적이다. 적당히 탐색을 했다면 더이상 무모한 모험을 떠나지 않아도 될 것이다. 반대로 한번도 당겨지지 않는 슬롯머신을 당겨볼 필요도 있다. 이런 문제점을 해결한 전략이 바로 <strong><u>UCB (Upper Confidence Bound)</u></strong>이다.<br>

위에서 언급한 알고리즘들은 보상(Reward)에만 초점을 뒀다면 UCB 알고리즘은 <strong><u>얼마나 많은 시행착오를 통해 알려진 정확한 결과</u></strong>인지도 함께 고려한다. 즉, 'A'가 가치투자를 하는 것이다! 다음과 같이 5가지의 슬롯머신이 결과값이 나왔다고 가정해보자.

* Total Try : 100
* A. 0.1412 (try: 5)
* B. 0.5423 (try: 61)
* C. 0.2134 (try: 8)
* D. 0.7864 (try: 24)
* E. 0.0013 (try: 2)

각 슬롯머신의 평균값을 보면 B 슬롯머신을 당기는 것이 무난해보인다.  100회 중 61회 시도하여 신뢰도도 높고 결과값도 준수하다. 하지만 D 슬롯머신을 보면 비록 24회 밖에 시도를 안해봤지만 결과가 훌륭하다. 우리는 어쩌면 B 슬롯머신보다 좋은 결과를 줄 수 있는 D 슬롯머신에 기대를 걸 필요가 있다. 이렇듯 UCB 알고리즘은 <strong><u>가치투자가 필요한 곳에 무게를 실어 「탐색(Exploration)」을 한다.</u></strong> 'A'는 UCB 알고리즘에 의하면 B 슬롯머신을 열심히 당기되(활용), 높은 확률로 D 슬롯머신을 당기는 「탐색(Exploration)」 을 할 것이다.

<br>
\\[
\Large
A_{t}\approx[Q_{t}(a)+c\sqrt{\frac{\log t}{N_{t}(a)}}]
\\]
<br>

여기서 $ c\sqrt{\frac{\log t}{N_{t}(a)}} $ 가 해당 슬롯머신이 최적의 슬롯머신이 될 수 있는 가능성을 뜻한다. $ c $ 는 𝜀-Greedy 전략의 𝜀 역할과 동일한 탐색의 정도를 조절하는 하이퍼파라미터이다. $ N_{t}(a) $ 는 해당 슬롯머신을 선택한 횟수로 선택됐을 수가 적으면 가중치를 주어 확률을 올린다.<br><br>

요약을 하자면 <strong><u>시도 횟수가 적은 슬롯머신은 꼼꼼하게, 시도 횟수가 많은 슬롯머신은 확신을 가진 뒤, 이 데이터를 바탕으로 최대의 값을 도출하는 알고리즘</u></strong>이다.<br><br>

아래는 UCB 알고리즘을 Python 언어를 가지고 구현을 해본 것이다.

{% highlight py %}

import numpy as np
import matplotlib.pyplot as plt

# 슬롯 머신 확률
def SlotData(num):
    # np.random.normal(수렴 값, 시그마(1))
    # (1) : 시그마 값이 클수록 랜덤값의 분포가 넓게퍼짐 -> 확률 저하
    if num == 1:
        return np.random.normal(0.5, 0.1)
    elif num == 2:
        return np.random.normal(0.6, 0.1)
    elif num == 3:
        return np.random.normal(0.7, 0.1)
    elif num == 4:
        return np.random.normal(0.3, 0.1)
    else:   # num == 5
        return np.random.normal(0.2, 0.1)

totalNumIteration = 5000    # 시도 횟수
numArm = 5                  # 슬롯 머신 개수

# Initalize
UCB = np.zeros((numArm, 1))                 # 각 슬롯머신의 UCB 값
numTry = np.zeros((numArm, 1))              # 각 슬롯머신의 시도 횟수
latest_reward = np.zeros((numArm, 1))       # 각 슬롯머신의 최근 값
average_reward = np.zeros((numArm, 1))      # 각 슬롯머신의 평균 값

# 그래프 만들기용 변수
# 모든 슬롯머신의 총 합의 평균 값 -> 모든 슬롯머신의 총합 / 슬롯머신의 개수
count_average_reward = np.zeros((1, totalNumIteration))

for i in range(1, totalNumIteration + 1):
    # 각 슬롯머신을 처음으로 당길 때
    if i <= numArm:
        latest_reward[i-1, 0] = SlotData(i)
        average_reward[i-1, 0] = latest_reward[i-1, 0]
        numTry[i-1] = 1
        
        if i == 1:
            count_average_reward[0 ,i-1] = latest_reward[i-1, 0]
        else:
            count_average_reward[0, i-1] = (count_average_reward[0, i-2]*(i-1) + latest_reward[i-1, 0]) / i
        
        if i == numArm:
            # UCB 알고리즘 수식
            UCB[:, 0] = average_reward[:, 0]  + np.sqrt((2 * np.log(i)) / numTry[:, 0])

    # 각 슬롯머신을 한번씩 다 당긴 후
    else:
        # selected_inx = UCB의 알고리즘에 의해 선택된 슬롯머신 넘버링
        selected_inx = np.argmax(UCB[:,0])
        latest_reward[selected_inx, 0] = SlotData(selected_inx+1)
        count_average_reward[0, i-1] = (count_average_reward[0, i-2]*(i-1) + latest_reward[selected_inx, 0]) / i
        numTry[selected_inx, 0] = numTry[selected_inx, 0] + 1
        average_reward[selected_inx, 0] = (average_reward[selected_inx, 0] * (numTry[selected_inx, 0] - 1) + latest_reward[selected_inx, 0]) / numTry[selected_inx, 0]
        # UCB 알고리즘 수식
        UCB[:, 0] = average_reward[:, 0] + np.sqrt((2 * np.log(i)) / numTry[:, 0])

.
.
.

{% endhighlight %}

<br>
![]({{site.baseurl}}/images/UCB_reward.png)
<br>

***

##### 마치며

사실 UCB 알고리즘보다 더 효과적인 「Thompson Sampling」 알고리즘이나  「Gradient Bandit」과 같은 여러 알고리즘이 존재한다. 나는 대표적으로 세가지 알고리즘만 공부하였다. 구글을 통해 여러 자료를 찾아보거나, 담당 교수님에게 조언을 구해 이론을 다지고, Python을 이용해 알고리즘을 구현하였다. 약 2주 가까이 공부를 하였고, 강화학습을 이해하는데 큰 도움이 되었다. 이제 이 알고리즘을 이용해 무언가 활용하는 방안을 생각해 봐야겠다. 아래는 내가 공부를 하는데 큰 도움이 된 사이트들이다.<br><br>

> 러닝머신의 Train Data set - [링크][link_2]<br>
> Chris-Song.brunch - [링크][link_3]<br>
> sanghyukchun.github.io - [링크][link_4]<br>
> 숨니의 무작정 따라하기 - [링크][link_5]

***

##### 출처

* [대문이미지] - [isdasquash.com][link_1]
* [1] - [러닝머신의 Train Data set][link_2]

[link_1]: http://www.isdasquash.com/casino-slot-machines/
[link_2]: https://m.blog.naver.com/nilsine11202/221912267111
[link_3]: https://brunch.co.kr/@chris-song/62
[link_4]: http://sanghyukchun.github.io/96/
[link_5]: https://sumniya.tistory.com/9