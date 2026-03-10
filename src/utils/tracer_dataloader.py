import torch
from torchvision import models
from decluttering.data.tracer.tusk_pipeline.tracer import Tracer, AnalyticTracer


class TracerDataloader():

    def __init__(self):
        self.tracer = Tracer()
        self.analytic_tracer = AnalyticTracer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # vit = models.vit_b_16(num_classes=2)
        # vit.load_state_dict(torch.load('/home/justinyu/vsumedh/src/best_model_945.pt'))
        # vit.to(self.device)
        # vit.eval()
        # self.vit_model = vit
        # self.tracer.vit_model = vit
