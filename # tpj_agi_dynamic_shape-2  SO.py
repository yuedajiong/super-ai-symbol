import torch
        
class DAT:
    def __init__(self, base, part, show, size=None):  #self._I_T = [[[0], [0,1]], [[5,4,3,2], [5,4,3]]];  for pair in self._I_T: yield pair[0], pair[1] if len(pair)>1 else None
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
            if size and idx==size: break
            if show: print(f'data load:  {idx:05d}/{len(full)}  {os.path.join(subs, item)} >>>')
            with open(os.path.join(subs, item), 'r') as file:
                name = item[:-5]
                text = json.load(file)
                data_ones = {}
                if part == 're-arc':
                    set_key = 'train'
                    set_dat = text
                    data_ones[set_key] = load_set(set_dat)
                else:
                    for set_key,set_dat in text.items():  #train  test
                        if set_key in ('train', 'test'):
                            data_ones[set_key] = load_set(set_dat)
                        else:
                            pass   #print(f'contain key: {set_key} in file: {os.path.join(subs, item)}, just pass for name, ...')
                buff[name] = data_ones 
        self.data = list(buff.values())                
    
    def __iter__(self):
        for pair in self.data:
            I, O = [x for row in pair['train'][0]['input'] for x in row], [x for row in pair['train'][0]['output'] for x in row]
            #print('vice_size', len(I), len(O))
            yield I, O

class OBJ:
    def __init__(self, form, vice):
        self.form = form
        self.vice = vice
        
    def __repr__(self):  #__str__
        return f"<form={self.form} vice={self.vice}>"             

class EMB(torch.nn.Module):  #John orthogonal for invertiable, not-learnable-parameters
    class Invertible(torch.nn.Module):
        def __init__(self, encode_len, encode_dim):
            super().__init__()
            self.encoder = torch.nn.Embedding(encode_len, encode_dim)
            self.decoder = torch.nn.Linear(encode_dim, encode_len, bias=False)
            for p in self.encoder.parameters(): p.requires_grad = False  
            for p in self.decoder.parameters(): p.requires_grad = False                
            with torch.no_grad(): torch.nn.init.orthogonal_(self.encoder.weight); self.encoder.weight /= self.encoder.weight.norm(dim=1, keepdim=True)  #John 正交归一eye
            self.decoder.weight = self.encoder.weight                                                                                                   #John 可逆权重共享
            
        def encode(self, D):
            return self.encoder(D)
            
        def decode(self, D):
            return self.decoder(D)
            
    def __init__(self, form_size, vice_size, hidden):
        super().__init__()
        self.form_inv = self.Invertible(form_size, hidden)
        self.vice_inv = self.Invertible(vice_size, hidden)
    
    def inflate(self, S, device):
        return OBJ(torch.tensor(S, dtype=torch.int64, device=device), torch.tensor([viceition for viceition in range(len(S))], dtype=torch.int64, device=device))

    def deflate(self, D, device='cpu'):
        return OBJ(form=D.form.argmax(dim=-1).to(device).tolist(), vice=D.vice.argmax(dim=-1).to(device).tolist())

    def encode(self, O):
        return OBJ(form=self.form_inv.encode(O.form), vice=self.vice_inv.encode(O.vice))
        
    def decode(self, L):          
        return OBJ(form=self.form_inv.decode(L.form), vice=self.vice_inv.decode(L.vice))

class TSA(torch.nn.Module):
    class DualAttention(torch.nn.Module):            
        class JointedBlend(torch.nn.Module):  #multi-head
            def __init__(self, hidden):
                super().__init__()  
                mix = hidden + hidden
                self.form_proj_k = torch.nn.Sequential(torch.nn.Linear(mix, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
                self.form_proj_v = torch.nn.Sequential(torch.nn.Linear(mix, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
                self.vice_proj_k = torch.nn.Sequential(torch.nn.Linear(mix, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
                self.vice_proj_v = torch.nn.Sequential(torch.nn.Linear(mix, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))                
                self.q_proj_form = torch.nn.Linear(hidden, hidden)
                self.q_proj_vice = torch.nn.Linear(hidden, hidden)
                self.o_proj_form = torch.nn.Sequential(torch.nn.Linear(hidden+hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
                self.o_proj_vice = torch.nn.Sequential(torch.nn.Linear(hidden+hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden))
            def forward(self, form, vice, old_form, old_vice):
                mix = torch.cat([form, vice], dim=-1)
                form_att_k, form_att_v, vice_att_k, vice_att_v = self.form_proj_k(mix), self.form_proj_v(mix), self.vice_proj_k(mix), self.vice_proj_v(mix)
                q_form = self.q_proj_form(old_form)
                a_form = torch.nn.functional.softmax((q_form @ vice_att_k.T) / (vice_att_k.size(-1) ** 0.5), dim=-1) @ vice_att_v
                q_vice = self.q_proj_vice(old_vice)
                a_vice = torch.nn.functional.softmax((q_vice @ form_att_k.T) / (form_att_k.size(-1) ** 0.5), dim=-1) @ form_att_v
                new_form = self.o_proj_form(torch.cat([old_form, a_form], dim=-1))
                new_vice = self.o_proj_vice(torch.cat([old_vice, a_vice], dim=-1))
                return new_form, new_vice 

        class DynamicShape(torch.nn.Module):
            def __init__(self, hidden, hidden_shape_factor):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(hidden, hidden*hidden_shape_factor), torch.nn.Tanh(), torch.nn.Linear(hidden*hidden_shape_factor, 1), torch.nn.Sigmoid())
            def forward(self, I):
                return self.net(I)                

        def __init__(self, hidden, hidden_shape_factor):
            super().__init__()
            self.hidden = hidden
            #self.head_route = ...       
            self.body_blend = self.JointedBlend(hidden)
            self.tail_shape = self.DynamicShape((hidden+hidden)*len(['old','new']), hidden_shape_factor=hidden_shape_factor)

        def forward(self, form, vice, shape_peak, shape_gate, shape_goal):    #TODO  结合路由，单个循环的不定长预测，需要逼出多步                       
            prob_halt_list = []
            outs_form, outs_vice = [], []
            old_form = torch.zeros(self.hidden, device=form.device)
            old_vice = torch.zeros(self.hidden, device=form.device)
            for shape_halt in range(0, shape_peak):  #TODO  latent compare based end, symbol compare based end
                new_form, new_vice = self.body_blend(form, vice, old_form, old_vice)
                outs_form.append(new_form.squeeze(0))
                outs_vice.append(new_vice.squeeze(0))
                
                prob_halt = self.tail_shape(torch.cat([old_form, old_vice, new_form, new_vice], dim=-1))  #TODO 看见什么东西做停止才是合理的？
                prob_halt_list.append(prob_halt)                
                if prob_halt.mean().item() > shape_gate:
                    if shape_goal is not None:
                        if shape_halt+1==shape_peak:
                            break
                    else:
                        break

                old_form, old_vice = new_form, new_vice
                
            loss_halt = 0
            if shape_goal is not None:  #TODO 用neural的方式去求action-space合适吗？ 如何做？
                prob_halt = torch.cat(prob_halt_list, dim=-1)
                goal_halt = torch.tensor([0 if i<shape_goal else 1 for i in range(prob_halt.size(0))], dtype=torch.float, device=form.device)
                print('prob_halt', prob_halt)
                print('goal_halt', goal_halt)
                print()
                loss_halt = torch.nn.functional.binary_cross_entropy(prob_halt, goal_halt, reduction='mean')             
                
            return torch.stack(outs_form, dim=0), torch.stack(outs_vice, dim=0), loss_halt, shape_halt+1
            
    def __init__(self, hidden, hidden_shape_factor):
        super().__init__()
        self.block = self.DualAttention(hidden, hidden_shape_factor)

    def forward(self, form_list, vice_list, shape_peak, shape_gate, shape_goal):  #Batch，I/T，理想Latent，要Gate+Loop
        form_outs, vice_outs, loss_halt_outs, size_halt_outs = [], [], [], []
        for batch, (form, vice) in enumerate(zip(form_list, vice_list)):  #batchify dynamic-shape  #mask=torch.where(cond, T, F)  
            form_out, vice_out, loss_halt, size_halt = self.block(form, vice, shape_peak=shape_peak, shape_gate=shape_gate, shape_goal=shape_goal[batch])
            form_outs.append(form_out)
            vice_outs.append(vice_out)
            loss_halt_outs.append(loss_halt)
            size_halt_outs.append(size_halt)        
        return form_outs, vice_outs, torch.stack(loss_halt_outs).mean(), size_halt_outs

class MSE:
    def forward(self, O, T, pad=0.0):
        def pad_mask_mse(o, t, pad):  #John if-pad-so-mask
            size = max(o.shape[0], t.shape[0])
            opad = torch.nn.functional.pad(o, (0, 0, 0, size - o.shape[0]), value=pad)
            tpad = torch.nn.functional.pad(t, (0, 0, 0, size - t.shape[0]), value=pad)
            loss = ((opad - tpad) ** 2).mean(dim=1)
            mask = torch.arange(size, device=o.device) < min(o.shape[0], t.shape[0])
            return (loss * mask).sum() / mask.sum()
        return sum(pad_mask_mse(o, t, pad=pad) for o, t in zip(O, T)) / len(O) 

def main(conf=type('',(),dict(hidden=32, hidden_shape_factor=16, shape_gate=0.5, epochs=1000, device=['cpu','cuda'][torch.cuda.is_available()]))()):
    def infer(perceptor, embedment, transform, shape_peak, shape_gate, device):
        latent = [[embedment.encode(embedment.inflate(i, device)), embedment.encode(embedment.inflate(t, device))] for (i,t) in perceptor]
        Fi = [l[0].form for l in latent]
        Vi = [l[0].vice for l in latent]
        Ft = [l[1].form for l in latent]        
        Vt = [l[1].vice for l in latent]       
        shape_goal=[ft.shape[0] for ft in Ft]
        Fo,Vo, loss_halt, size_halt = transform(Fi, Vi, shape_goal=shape_goal, shape_peak=shape_peak, shape_gate=conf.shape_gate)  
        return latent, Ft,Fo, Vt,Vo, shape_goal, loss_halt, size_halt
    
    perceptor = DAT(base='./data/arcagi1/', part='training', show=0, size=1)
    vice_size = 81
    shape_peak = 99
    
    embedment = EMB(form_size=12, vice_size=vice_size, hidden=conf.hidden).to(conf.device)
    transform = TSA(hidden=conf.hidden, hidden_shape_factor=conf.hidden_shape_factor).to(conf.device)
    criterion = MSE()
    
    optimizer = torch.optim.Adam([{'params':transform.parameters(), 'lr':0.001}], lr=0.001)  #{'params':embedment.parameters(), 'lr':0.0} 
  
    transform.train()
    for epoch in range(conf.epochs+1):
        latent, Ft,Fo, Vt,Vo, shape_goal, loss_halt, size_halt = infer(perceptor, embedment, transform, shape_peak=shape_peak, shape_gate=conf.shape_gate, device=conf.device)
       
        loss_F = criterion.forward(Fo, Ft)
        loss_V = criterion.forward(Vo, Vt)
        loss = (1.0*loss_halt) + (1.0*loss_F) + (1.0*loss_V)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%10==0: print(f'epoch={epoch:04d}/{conf.epochs}  loss={loss.item():7.4f}  F:{loss_F.item():.4f} + V:{loss_V.item():.4f} + H:{loss_halt.item():.4f}  size:{list(size_halt)} / goal:{list(shape_goal)}')

    transform.eval()
    with torch.no_grad():
        latent, Ft,Fo, Vt,Vo, shape_goal, loss_halt, size_halt = infer(perceptor, embedment, transform, shape_peak=shape_peak, shape_gate=shape_gate, device=conf.device) 
       
        Di = [[i,t] for (i,t) in perceptor]
        print('Di', Di)
       
        Si = [[embedment.deflate(embedment.decode(l[0]))] for l in latent]
        print('Si', Si)
        St = [[embedment.deflate(embedment.decode(l[1]))] for l in latent]
        print('St', St)        

        Lo = [OBJ(f, v) for f,v in zip(Fo,Vo)]
        So = [[embedment.deflate(embedment.decode(l))] for l in Lo]
        print('So', So) 

if __name__ == "__main__":
    main()

