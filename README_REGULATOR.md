# MPC Regulator Mode Implementation

## 概述

本实现为Crazyflie仿真系统添加了一种新的MPC控制模式——**Regulator模式**。在这种模式下，每一步的路径追踪都被转换为"回归到原点"的问题，与传统的轨迹追踪模式形成对比。

## 核心理念

### 传统Tracking模式
- 使用全局轨迹作为参考
- MPC预测窗口包含未来多个轨迹点
- 直接优化跟踪全局路径

### 新的Regulator模式  
- 每个时间步将目标点转换为相对于当前位置的局部坐标系
- 在局部坐标系中，目标始终是"原点"
- 更类似于传统的调节器控制

## 实现详情

### 1. 核心组件

#### 新增的枚举和类
```python
class ControlMode(Enum):
    TRACKING = "tracking"     # 传统轨迹追踪
    REGULATOR = "regulator"   # 回归到原点的调节器模式

class RegulatorMPCSimulator(SimpleMPCSimulator):
    """MPC simulator in regulator mode"""
```

#### 工厂函数
```python
def create_simulator(problem: Dict):
    """根据控制模式自动选择合适的模拟器"""
```

### 2. 文件修改

#### `tinympc_generator.py`
- ✅ 添加了`ControlMode`枚举
- ✅ 扩展`generate_problem()`方法支持控制模式参数
- ✅ 实现`RegulatorMPCSimulator`类
- ✅ 添加`create_simulator()`工厂函数

#### `generate.py`
- ✅ 添加了`--mode`命令行选项
- ✅ 支持选择tracking或regulator模式

#### `parameter_study.py`
- ✅ 添加了`--control-mode`命令行选项
- ✅ 所有分析功能支持两种控制模式

### 3. 新增测试和分析工具

#### `test_regulator_mode.py`
- ✅ 全面的验证测试套件
- ✅ 性能比较和可视化
- ✅ 多轨迹类型测试

#### `mode_analysis.py`
- ✅ 高级分析工具
- ✅ 统计显著性检验
- ✅ 噪声敏感性分析
- ✅ 参数敏感性分析

#### `demo_regulator.py`
- ✅ 简单演示脚本
- ✅ 快速上手指南

## 使用方法

### 1. 基本使用

```bash
# 使用tracking模式（默认）
python generate.py circle

# 使用regulator模式
python generate.py circle --mode regulator

# 参数研究
python parameter_study.py --control-mode regulator --trajectory circle
```

### 2. 程序化使用

```python
from tinympc_generator import TinyMPCGenerator, TrajectoryType, ControlMode, create_simulator

# 创建生成器
generator = TinyMPCGenerator()

# 生成regulator模式问题
problem = generator.generate_problem(
    control_freq=50.0,
    horizon=30,
    traj_type=TrajectoryType.CIRCLE,
    traj_duration=10.0,
    control_mode=ControlMode.REGULATOR,  # 新的参数
    radius=1.0,
    center=[0, 0, 1.5]
)

# 创建合适的模拟器
simulator = create_simulator(problem)

# 运行仿真
simulator.simulate(steps=500)
```

### 3. 测试和验证

```bash
# 运行验证测试
python test_regulator_mode.py

# 运行全面分析
python mode_analysis.py --quick

# 运行演示
python demo_regulator.py
```

## 技术细节

### Regulator模式的实现逻辑

1. **参考轨迹生成**: 在每个时间步，将全局目标点转换为相对于当前状态的轨迹
2. **MPC求解**: 在局部坐标系中求解优化问题
3. **控制应用**: 将计算出的控制输入应用到实际系统

### 关键算法

```python
def _generate_regulator_references(self, step: int):
    """生成regulator模式的参考轨迹"""
    target_state = self.X_ref[:, current_step]
    
    # 生成从当前状态到目标的平滑轨迹
    for i in range(self.horizon):
        alpha = min(i / (self.horizon - 1), 1.0)
        X_ref_horizon[:, i] = (1 - alpha) * self.x_current + alpha * target_state
    
    return X_ref_horizon, U_ref_horizon
```

## 性能特征

### 预期差异

1. **收敛性**: Regulator模式可能在某些条件下具有更好的收敛特性
2. **鲁棒性**: 对于噪声和扰动的敏感性可能不同
3. **计算效率**: 两种模式的计算复杂度基本相同
4. **跟踪精度**: 在不同轨迹类型下表现可能有差异

### 适用场景

- **Tracking模式**: 适用于精确轨迹跟踪，如表演飞行、测量任务
- **Regulator模式**: 适用于点到点导航、避障控制、稳定性要求高的应用

## 向后兼容性

- ✅ 所有现有功能保持不变
- ✅ 默认行为为传统tracking模式
- ✅ 现有脚本无需修改即可继续使用
- ✅ 新功能通过可选参数提供

## 扩展性

该实现框架支持进一步扩展：

1. **新的控制模式**: 可以轻松添加其他控制策略
2. **自适应切换**: 可以实现动态模式切换
3. **参数调优**: 可以为不同模式优化不同的代价函数权重

## 总结

这次实现成功地将MPC regulator方法集成到了现有的Crazyflie仿真系统中，提供了：

- 🔧 **完整的实现**: 从核心算法到用户接口
- 📊 **全面的测试**: 验证和性能分析工具
- 🎯 **保持兼容**: 不破坏现有功能
- 📚 **文档完善**: 使用指南和技术说明

用户现在可以选择使用传统的轨迹追踪模式或新的regulator模式，并通过提供的分析工具比较它们在不同条件下的性能。