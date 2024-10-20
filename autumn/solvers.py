def euler_step(get_derivative, take_step, position_0):
    return get_derivative(position_0)

def heun_step(get_derivative, take_step, position_0):
    direction_0 = get_derivative(position_0)
    position_1 = take_step(position_0, direction_0, 0, 1)
    direction_1 = get_derivative(position_1, 1)
    return (direction_0 + direction_1) / 2

def rk2_step(get_derivative, take_step, position_0):
    direction_0 = get_derivative(position_0)
    position_1 = take_step(position_0, direction_0, 0, 0.5)
    return get_derivative(position_1, 0.5)
    
def rk4_step(get_derivative, take_step, position_0):
    direction_0 = get_derivative(position_0)
    position_1 = take_step(position_0, direction_0, 0, 0.5)
    direction_1 = get_derivative(position_1, 0.5)
    position_2 = take_step(position_0, direction_1, 0, 0.5)
    direction_2 = get_derivative(position_2, 0.5)
    position_3 = take_step(position_0, direction_2, 0, 1)
    direction_3 = get_derivative(position_3, 1)
    return (direction_0 + 2 * (direction_1 + direction_2) + direction_3) / 6
