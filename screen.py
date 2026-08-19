import pyautogui


def capture_screen():

    screenshot = pyautogui.screenshot()

    path = "current_screen.png"

    screenshot.save(path)

    return path