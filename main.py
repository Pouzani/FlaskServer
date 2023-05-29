import torch
from torchvision.transforms import transforms
from PIL import Image
from flask import Flask, request

from resnet9 import ResNet9

# Création de l'instance du modèle ResNet9
model = ResNet9(in_channels=3, num_diseases=10)
model.load_state_dict(ResNet9.load_state_dict('IntelliFarm.pth'))

# Chargement des poids du modèle à partir du fichier .pth
model.load_state_dict(torch.load('IntelliFarm.pth', map_location=torch.device('cpu')))
model.eval()

# Création de l'instance de l'application Flask
app = Flask(__name__)

# Définition de la route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Récupération de l'image envoyée dans la requête
    image_file = request.files['image']
    image = Image.open(image_file)

    # Transformation de l'image pour la passer en entrée du modèle
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Prédiction de la classe de la maladie
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()

    # Retour de la classe prédite dans la réponse de la requête
    return str(predicted_class)

# Exécution de l'application Flask
if __name__ == '__main__':
    app.run()