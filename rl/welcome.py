from env.foo import getHello

class WelcomeManager():
    def welcome(self):
        message = getHello()
        print(message + " World!!!")

    def bye(self):
        print("Bye!")
        return "Done"
