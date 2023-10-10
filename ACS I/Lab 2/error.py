def errrorHandler(e):
    try:
        e()
    except:
        print("Something went wrong")

def sum():
    a = 1
    raise Exception("From sum")
    b = 2
    sum = a + b


def square():
    a = 1
    b = 2
    sum = a**b

def main():
    errrorHandler(sum)
    errrorHandler(square)

main()