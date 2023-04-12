from gui import FringesGUI


def main():
    gui = FringesGUI()  # todo: encapsulate this with a with statement to ensure params get saved upon shutdown and/or crash
    gui.show()


if __name__ == "__main__":
    main()
