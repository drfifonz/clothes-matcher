import argparse

from os import get_terminal_size

RED = "\033[31m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0;0m"


def arguments_parser():
    """
    set arguments and options to runnig script
    """
    parser = argparse.ArgumentParser(description="Clothes matcher")

    parser.add_argument("--type", type=str, default="train", help="[test/train]")

    return parser.parse_args


def print_all_user_arguments(arguments: argparse.Namespace) -> None:
    """
    Function prints all values setted in parser
    """

    terminal_size = get_terminal_size()

    print("\u2500" * terminal_size.columns)

    for argument in vars(arguments):
        value = getattr(arguments, argument)

        if isinstance(value, bool) and value is True:
            value = GREEN + str(value) + RESET_COLOR
        elif isinstance(value, bool) and value is not True:
            value = RED + str(value) + RESET_COLOR

        print(f"{argument.upper()} = {value}")

    print("\u2500" * terminal_size.columns)
