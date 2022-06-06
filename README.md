# Mediapipe-HandsTracker的TFLite模型Python推理接口

#### 更新
- [x] Mediapipe TFLite的初步逻辑完成(2021.01.07)
- [x] 加入基于规则的单手手势识别(2021.01.14)
- [x] 支持手检测器输入(2022.01.28)
- [x] 支持双手逻辑(2022.02.06)
- [x] 支持肢体关键点输入(2022.02.08)

#### 使用WebCam-双手逻辑测试
```
python scripts/demo_hands.py --camera --debug[optional] --roi_mode=[0|1]
# debug: 显示额外信息;
# roi_mode: 0为手检测器输入，1为肢体关键点输入;
# capability: 0为关键点小模型，1为大模型;
```

##
#### 默认读取videos目录下的hand_test_02.mp4作为测试视频
```
python scripts/demo.py
# inp: 可通过parser的inp修改测试视频；
```

#### 使用WebCam测试
```
python scripts/demo.py --camera
# roi: 默认是使用检测器，选择roi会用默认roi区域作为第一帧输入
```

#### 显示World-Landmarks
```
python scripts/demo.py --camera --draw3d
```

#### TFLite Full模型与Lite模型切换
```
python scripts/demo.py --capability=1
# capability: 1代表初始化Full模型，0代表使用Lite模型；
```

#### 说明：
- TFLite的Full & Lite模型以及MNN的Full & Lite模型都在lib/models目录下；
- Parser参数有pipe_mode选项，**1**代表使用Mediapipe的**旋转转正链路**，**0**代表使用我们之前的**简单逻辑链路**，默认是使用Mediapipe的数据链路；
- 测试过程自动保存在save目录下；
- Demo仅支持单手，通过设置--roi可以不使用检测器，使用预设置的roi，第一帧对**红色框区域**作为关键点模型的输入，后续使用前一帧关键点结果计算ROI作为下一帧输入；
- 左上角**红色概率值**代表手在ROI里面的概率，也用于下一帧用不用前一帧的关键点的依据（概率阈值为0.5）；
- 左上角**LeftHand or RightHand**是手的左右手分类结果；
- 左下角**红色Input区域**表示通过**旋转转正链路**旋转后输入给关键点模型的输入ROI区域；
- **蓝色旋转框**表示根据前一帧关键点结果计算的ROI区域；
- 部分函数是基于Mediapipe的C++代码转python的，因此此类函数部分也包含有**C++代码**，可与python代码一起参考，相信会有帮助；
- 关键点周围圆圈颜色代表关键点离相机的远近距离，离相机近的关键点周围变成白色，远的关键点周围圆圈变黑色，这个颜色只能用于判断哪个关键点更接近相机，无法用于计算距离；
