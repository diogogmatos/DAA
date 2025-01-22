from simple_term_menu import TerminalMenu


def read():
    try:
        with open("data/settings.txt", "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        with open("data/settings.txt", "x") as file:
            file.write()
            return []
        

SELECTED = read()
        

def write(selected):
    global SELECTED
    with open("data/settings.txt", "w") as file:
        for selection in selected:
            file.write(f"{selection}\n")
    SELECTED = selected


def show():
    global SELECTED
    SELECTED = read()
    options = ["RFECV", "SMOTE", "PCA", "Use cached RFECV selection data.", "Use cached estimator data."]
    menu = TerminalMenu(options, multi_select=True, multi_select_empty_ok=True, preselected_entries=SELECTED, show_multi_select_hint_text="Press SPACE to select, ENTER to confirm.", show_multi_select_hint=True, title="Settings", multi_select_select_on_accept=False)
    menu.show()
    selected = []
    if menu.chosen_menu_entries:
        selected = [x for x in menu.chosen_menu_entries]
    write(selected)