
def interpolation(eye: float, cal_screen1: int, cal_screen2: int, cal_eye1: float, cal_eye2: float) -> float:
    return cal_screen1 + (eye - cal_eye1) * (cal_screen2 - cal_screen1) / (cal_eye2 - cal_eye1)


def geometric_algorithm(eye: float, deltaR: float, distance2screen: float, r_eyeball: float, k: float) -> float:
    return k * ((distance2screen + r_eyeball) * (eye / r_eyeball) + deltaR)


# Gaze mapping method1
def linear_mapping(eye: (float, float), cal_screen1: (int, int), cal_screen2: (int, int), cal_eye1: (float, float),
                   cal_eye2: (float, float)) -> (float, float):

    x = interpolation(eye[0], cal_screen1[0], cal_screen2[0], cal_eye1[0], cal_eye2[0])
    y = interpolation(eye[1], cal_screen1[1], cal_screen2[1], cal_eye1[1], cal_eye2[1])

    return x, y


# Gaze mapping method2
def second_order_mapping(cal_screen: list, eye: (float, float)) -> (float, float):

    x = cal_screen[0][0] + cal_screen[0][1] * eye[0] + cal_screen[0][2] * eye[1] + cal_screen[0][3] * eye[0] * eye[1]
    y = cal_screen[1][0] + cal_screen[1][1] * eye[0] + cal_screen[1][2] * eye[1] + cal_screen[1][3] * eye[1] * eye[1]

    return x, y


# Gaze mapping method3
def geometric_mapping(eye: (float, float), deltaR: float, distance2screen: float, r_eyeball: float, k: float) -> (float, float):

    x = geometric_algorithm(eye[0], deltaR, distance2screen, r_eyeball, k)
    y = geometric_algorithm(eye[1], deltaR, distance2screen, r_eyeball, k)

    return x, y
