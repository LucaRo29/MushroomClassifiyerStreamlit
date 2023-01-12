from lib import Flask, np, transforms, torch, num_classes, MODEL_PATH


class MushroomClassifier(Flask):
    def __init__(self, import_name: str):
        super().__init__(import_name)
        self.config['UPLOAD_FOLDER'] = './static/uploads/'
        self.secret_key = 'super secret key'
        self.config['SESSION_TYPE'] = 'filesystem'
        self.crop_transformations = None
        self.norm_transformations = None
        self.shroom_model = None
        self.initialize_model()

    def initialize_model(self):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_size = (224, 224)

        self.crop_transformations = transforms.Compose([
            transforms.CenterCrop(600),
            transforms.Resize(img_size)
        ])
        self.norm_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.shroom_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights="ResNet18_Weights.DEFAULT")
        num_features = self.shroom_model.fc.in_features
        self.shroom_model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )
        self.shroom_model.load_state_dict(torch.load(MODEL_PATH))
        self.shroom_model.eval()
