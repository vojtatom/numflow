import sys
from .write import printOK


from .load import load, sample_rectilinear_dataset
from .exception import NumflowException

class Application:
    def __init__(self):
        self.dataset = None
        self.commands = {
            'open': self.open
        }

    def process(self, data):
        printOK("app got {}".format(data))
        
        try:
            #there has to be a command in the data dictionary
            if 'command' not in data:
                return {'status': 'error'}

            if data['command'] in self.commands:
                return self.commands[data['command']](data)

        except NumflowException as e:
            return {'status': 'error',
                    'message': e.title }
        except e:
            return {'status': 'error',
                    'message': str(e) } 

    def open(self, data):
        #try opening the new dataset
        filename = data['filename']
        self.dataset = load(filename, mode='c')

        print('loaded')
        data = sample_rectilinear_dataset(self.dataset, 16, 16, 16)

        
        return {'status': 'okay',
                'message': 'dataset loaded'}

    