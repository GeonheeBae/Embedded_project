import math
ki = 0; kp = 2; kd = 0.0

#PID 제어기 설계

def steering_input(error, error_prev, error_accum, Ts): 
    
    # 에러가 양수일때 오른쪽 선 / 오른쪽으로 돌아야 함
    # 전 에러보다 현 에러가 더 오른쪽(양수)이면 더 꺾어야 하는데 d 제어기로 이를 막아야함
    # error - error_prev 이게 음수여야함    
    # 배건희
    error_accum = error_accum + error*Ts 
    
    # integrate windup
    reset_scale = 0.2
    if error_accum*ki >= reset_scale:
        error_accum = reset_scale/ki
    elif error_accum*ki <= -reset_scale:
        error_accum = -reset_scale/ki

    error_diff = (error - error_prev)/Ts 
    tau = 1
    steering_input = ki*error_accum + kp*error + kd*(error_diff/(tau*error_diff + 1))

    
    return [steering_input, error_accum, error, error_diff]