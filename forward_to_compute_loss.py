from data_generator import DatasetLoader
import torch  
from tqdm import tqdm
from models import cost_fn
import pandas as pd
from models.mbnetv2 import MobileNetV2
# Generate Data
sample_path = "dataset/last_face_age_gender"
train_loader = DatasetLoader(sample_path, "train")   
val_loader = DatasetLoader(sample_path, "val")

params = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 8
}

train_generator = torch.utils.data.DataLoader(train_loader, **params)
val_generator = torch.utils.data.DataLoader(val_loader, **params)
dataset = (
    ("train", train_generator),
    ("val", val_generator)

)

# Create csv data
csv_dict = {
    "Sample": [],
    "Age Loss": [],
    "Gender Loss": [],
    "Train Loss": [],   
    "Val Loss": []
}

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model 
AGE_CLASSIFIER = True 
GENDER_CLASSIFIER = True
model = MobileNetV2(
    widen_factor=1.0, 
    num_age_classes=16,      
    num_gender_classes=2   
)

model = model.to(device)
model.load_state_dict(torch.load("19_0.9276765206149805_gender_1.0798624753952026_age.pt"))
model.eval()



val_loss, train_loss, loss_age, loss_gender = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

for category, data in dataset:
    with torch.no_grad():
        for path, image, (label_age, label_gender) in tqdm(data, desc=f"{category} process"):
            image = image.to(device)
            label_age = label_age.to(device)
            label_gender = label_gender.to(device)
            
            output = model(image)

            if AGE_CLASSIFIER and GENDER_CLASSIFIER:
                score_age, pred_age, pred_gender = output
                loss_age = cost_fn.cost_nll(pred_age, label_age)        
                loss_gender = cost_fn.cost_nll(pred_gender, label_gender)     
                loss = loss_age + loss_gender

            elif AGE_CLASSIFIER and not GENDER_CLASSIFIER:
                score_age, pred_age = output 
                loss_age = cost_fn.cost_nll(pred_age, label_age)                    
                loss = loss_age
            else:
                pred_gender = output
                loss_gender = cost_fn.cost_nll(pred_gender, label_gender)
                loss = loss_gender
            
            csv_dict["Sample"].append(path)
            csv_dict["Age Loss"].append(loss_age.item())
            csv_dict["Gender Loss"].append(loss_gender.item()) 
            if category == "train":
                csv_dict["Train Loss"].append(loss.item())
            else:
                csv_dict["Val Loss"].append(loss.item())
    
            df = pd.DataFrame(csv_dict)
            df.to_csv("loss_debuging.csv")
print("DONE!")

        