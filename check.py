import torch


dict1= {
    "a": 1,
    "b": 2,
    "c": 3
}

dict2 = {"d": 4, "e": 5}

dict3 = {"f": 6, "g": 7}

def check_some(**kwargs):
    kwargs = {**kwargs}
    print("Received arguments:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)


def check_2(*args):
    for arg in args:
        print(f"Argument: {arg}")


if __name__ == "__main__":
    print(("d","e") in dict2.items())