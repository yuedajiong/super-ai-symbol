class Engine:
    def __init__(self):
        self.methods = {}
        self.tracing = []

    def _register(self, name, method):
        print(f"register {name} with {method}")
        self.methods[name] = method

    def _dispatch(self, name, context):
        self.tracing.append(f"dispatch {name} with {context}")
        result = self.methods[name](context)
        context[name] = context
        return result

    def loop(self, method, inputs, max_steps):
        def register_class(flag, clas):
            for name in dir(clas):
                attr = getattr(clas, name)
                if not name.startswith("_") and callable(attr): self._register((flag, name), attr)
        for name in method.__dict__:
            if not name.startswith("_"):
                attr = getattr(method, name)
                if isinstance(attr, type):
                    if name == 'State': register_class('S', attr)
                    elif name == 'Action': register_class('A', attr)
                    else: raise 
                elif callable(attr):
                    if name == 'coord': self._register((' ',name), attr)
                    elif name == 'route': self._register((' ',name), attr)
                    elif name == 'judge': self._register((' ',name), attr)
                    else: raise                
                else:
                    raise 

        context = {'inputs':inputs}
        for _ in range(max_steps):
            state = self._dispatch((' ','coord'), context)
            route = self._dispatch(('S', state), context)
            action = self._dispatch((' ','route'), context) 
            judge = self._dispatch(('A', action), context)
            finish = self._dispatch((' ','judge'), context)
            if finish: break

        for step,line in enumerate(self.tracing): print(f'trace:  {step:04d}  {line}')

import random
class Method:
    def coord(context):  #neural(context) -> state
        I = context['inputs']['I']
        if I%2==0:
            state = 'extract_shape'
        else:
            state = 'extract_color'
        return state

    class State:         #neural(context)
        def extract_shape(context):
            context['state'] = {'route':'shape'}
            context['state']['shape'] = random.choice([0, 1, 2])

        def extract_color(context):
            context['state'] = {'route':'color'}
            context['state']['color'] = random.choice([0, 1])

    def route(context):  #neural(context) -> action
        if context['state']['route'] == 'shape':
            if context['state']['shape'] == 0:
                return 'execute_trans'
            elif context['state']['shape'] == 1:
                return 'execute_rotat'
            elif context['state']['shape'] == 2:
                return 'execute_scale'
            else: raise
        elif context['state']['route'] == 'color':
            if context['state']['color'] == 0:
                return 'execute_black'
            elif context['state']['color'] == 1:
                return 'execute_white'
            else: raise
        else:
            raise

    class Action:        #symbol(context)
        def execute_trans(context):
            context['action'] = 'trans'

        def execute_rotat(context):
            context['action'] = 'rotat'

        def execute_scale(context):
            context['action'] = 'scale'

        def execute_black(context):
            context['action'] = 'black'

        def execute_white(context):
            context['action'] = 'white'

    def judge(context):  #manual(context) -> stopped
        stopped = context['state']['route'] == 'color' and context['action']=='white'
        print('$$$$' if stopped else '####', '  ', context['state']['route'], context['action'])
        return stopped

def main():
    Engine().loop(method=Method, inputs={'I':random.choice([0, 1])}, max_steps=2)

if __name__ == '__main__':
    main()

