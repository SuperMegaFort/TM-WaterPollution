import matplotlib.pyplot as plt
from PIL import Image

def verifier_resolution(image_path):
    print(f"Ouverture de l'image : {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
    

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(img)
    axes[0].set_title(f"1. Originale ({width}x{height})")
    axes[0].axis('off')
    
    axes[1].imshow(img_resized)
    axes[1].set_title("2. Redimensionnée 224x224\n(Perte de détails, ce que voit l'IA)")
    axes[1].axis('off')
    
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Remplace par le chemin de ton image test qui contient de l'irisation
    # Par exemple, celle que tu m'as envoyée :
    CHEMIN_IMAGE = "dataset/val/4/24032021_185000_RCNX0667_Aire.jpg" 
    
    verifier_resolution(CHEMIN_IMAGE)