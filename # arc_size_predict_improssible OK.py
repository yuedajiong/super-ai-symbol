import torch

class Data:  #(torch.utils.data.Dataset)
    def __init__(self, base, part, show, size=None, skip=[]):
        def load_set(set_dat):
            data_list = []
            for set_one in set_dat:
                data_dict = {}
                for pari_key,pair_dat in set_one.items():  #input  output
                    data_dict[pari_key] = pair_dat
                data_list.append(data_dict)
            return data_list
        buff = {}
        import os, json
        subs = os.path.join(base, part)
        full = sorted(os.listdir(subs))
        for idx,item in enumerate(full):
            if size and idx==size: break  #John
            name = item[:-5]
            if skip and name in skip: continue
            if show: print(f'data load:  {idx:05d}/{len(full)}  {os.path.join(subs, item)} >>>')
            with open(os.path.join(subs, item), 'r') as file:
                
                text = json.load(file)
                data_ones = {}
                if part == 're-arc':
                    set_key = 'train'
                    set_dat = text
                    data_ones[set_key] = load_set(set_dat)
                else:
                    for set_key,set_dat in text.items():
                        if set_key in ('train', 'test'):
                            data_ones[set_key] = load_set(set_dat)
                        else:
                            if show: print(f'contain key: {set_key} in file: {os.path.join(subs, item)}, just pass for name:', name)
                buff[name] = data_ones
        self.data = list(buff.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Call:  #function-as-parameter
    #mine: data: NN                         #supervised
    #  nn(.)
    #skim: NN -> OPs                        #john-new-algorithm    #remeber-to-compose;  parameter-freeze;  trajectory-route;  
    #  nn(.) -> OP_?(ψ, .)
    #  nn(.) -> OP_?(ψ, .)
    #pick: clue: θ, I, T  -> OP_?(ψ, .)     #reinforce, search     #I:grid-size, color-number, object-number(eb5a1d5d), picked-object-size
    #  OP_?(ψ, θ,I) -> I    <--> T
    #  OP_?(ψ, θ,I) -> I*2  <--> T
    #  OP_?(ψ, θ,I) -> I^2  <--> T
    #  OP_?(ψ, θ,I) -> ?    <--> T          #john-pit              #generality
    #exec: quiz: θ, I, OP_? -> O
    #  OP_?(ψ, θ,I) -> 
 
    def indentity(i):
        return i

class Mind(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, i):
        pass

class Size:  #soso
    def __init__(self, data, cheat=0, show=0):
        self.testo = []
        sizes = []
        shots = len(data['train'])
        for set_key in ('train','test'):
            for set_one in data[set_key]:
                ipt = set_one['input']
                opt = set_one['output'] if 'output' in set_one else None
                sizes.append([[len(ipt),len(ipt[0])], [len(opt),len(opt[0])]])
        if all(tuple(ipt) == tuple(opt) for ipt, opt in sizes[:shots]):
            if show: print('i_o_same_size in train')
            for size in sizes[shots:]:
                self.testo.append(size[0])
        elif len({tuple(size[1]) for size in sizes if size[1]}) == 1:
            if show: print('*_o_same_size if output')
            for size in sizes[shots:]:
                self.testo.append(sizes[0][1])
        elif len({tuple(size[0]) for size in sizes}) == 1:
            if show: print('*_i_same_size at all')
            max_x, max_y = 0, 0
            for size in sizes:
                for io in size:
                    if io: max_x, max_y = max(max_x, io[0]), max(max_y, io[1])
            for size in sizes[shots:]:
                self.testo.append([max_x, max_y])
        else:
            if cheat:
                for size in sizes[shots:]:
                    self.testo.append(size[1] if size[1] else [33, 34])
            else:
                for size in sizes[shots:]:
                    self.testo.append([34, 34])

    def __call__(self):
        return self.testo

def main(cfg=type('',(),dict(device=['cpu','cuda'][torch.cuda.is_available()]))()):
    data = Data(base='./data/arcagi1/', part='training', show=0, size=None, skip=[])
    for i in range(len(data)):
        name, item = data[i]
        testo = Size(item)()
        testt = [[len(x['output']),len(x['output'][0])] for x in item['test']]
        if testo==testt:
            print('ok $$', '', name, '', testt, '==' ,testo)
        else:
            print('NO xx', '', name, '', testt, '!=' ,testo)
            def stat(grid):                
                return len(set(one for row in grid for one in row))  #{one: sum(row.count(one) for row in arr) for one in set(one for row in x['input'] for one in row)}
            for x in item['train']:
                print('                   !:', 'size_i:',[len(x['input']),len(x['input'][0])], '->', 'size_o:',[len(x['output']),len(x['output'][0])], '\t', 'color_i:', stat(x['input']))
            for x in item['test']:
                print('                   ?:', 'size_i:',[len(x['input']),len(x['input'][0])], '->', 'size_o:',[len(x['output']),len(x['output'][0])], '\t', 'color_i:', stat(x['input']))
if __name__ == '__main__': 
    main()

