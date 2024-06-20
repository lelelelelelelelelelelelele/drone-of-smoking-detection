class PIDctl:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.setpoint = 0  # 设定点
        self.current_value = 0  # 当前反馈值
        self.integral = 0  # 积分项
        self.derivative = 0  # 微分项
        self.prev_error = 0  # 上一时刻误差
        self.dt=dt

    def update(self, error):
        #error = target - current  # 当前误差

        # 计算积分项（需要考虑积分饱和防止积分爆炸）
        self.integral += error * self.dt  # dt 是采样周期

        # 计算微分项
        #self.derivative = (error - self.prev_error) / dt

        # 计算PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * self.derivative

        # 更新上一时刻误差
        self.prev_error = error

        return output
    # 在某些特定条件满足时，可以调用此函数重置积分项
    def reset_integral(self):
        self.integral = 0
        self.prev_error = 0
'''
class tst:
    def __init__(self):
        self.i=0
    def add(self):
        self.i+=1
        return self.i
    
tt=tst()
tt.add()
tt.add()
tt.add()
print(tt.add()) '''