from typing import List,Dict,Tuple

class agent_A():

    def __init__(self,a,b) -> None:
        self.a = a
        self.b = b
    
    def have_info(self):

        x = self.a
        y = self.b
        have_info = x,y

        return have_info
    
    def send_info(self):
        info = 'do_something_here'
        return info
    
class agent_B():

    def __init__(self) -> None:
        pass

    def recieve_info():

        pass

    def process_info():
        x = "add some thing info"
        return x
    
if __name__ == "__main__":

    obj = agent_A('hello','hello')
    mid = obj.send_info()
    final = agent_B.recieve_info()
    final = agent_B.process_info()


        