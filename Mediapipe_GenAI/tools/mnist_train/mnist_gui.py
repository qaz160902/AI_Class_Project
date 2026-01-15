import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# --- 1. 模型架構 (需與訓練時完全一致) ---
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. GUI 應用程式類別 ---
class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手寫數字辨識 (MNIST)")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        # 設定裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 載入模型
        self.model = ConvNet().to(self.device)
        # 使用相對於程式檔案的路徑，確保無論在哪執行都能找到
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'mnist_cnn.pth'))
        
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("模型載入成功！")
        except FileNotFoundError:
            messagebox.showerror("錯誤", f"找不到模型檔案：\n{self.model_path}\n請確認模型位置。")
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("錯誤", f"載入模型失敗：\n{e}")
            self.root.destroy()
            return

        # 定義圖片轉換 (與訓練時一致，加上 Resize 和 Grayscale)
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 建立 UI 元件
        self.create_widgets()

    def create_widgets(self):
        # 標題
        tk.Label(self.root, text="請上傳手寫數字圖片", font=("Arial", 16)).pack(pady=20)

        # 圖片顯示區域 (Canvas)
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="lightgray")
        self.canvas.pack(pady=10)

        # 按鈕區域
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        btn_upload = tk.Button(btn_frame, text="選擇圖片", command=self.upload_image, font=("Arial", 12))
        btn_upload.pack(side=tk.LEFT, padx=10)

        # 結果顯示
        self.lbl_result = tk.Label(self.root, text="預測結果：---", font=("Arial", 18, "bold"), fg="blue")
        self.lbl_result.pack(pady=10)
        
        self.lbl_prob = tk.Label(self.root, text="信心度：---", font=("Arial", 10))
        self.lbl_prob.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        try:
            # 讀取並顯示原始圖片
            img = Image.open(file_path).convert("L")  # 轉為灰階
            
            # 處理顏色反轉 (如果是白底黑字，要轉成黑底白字)
            # 簡單判斷：如果平均亮度 > 127，假設是白底，進行反轉
            stat = ImageOps.grayscale(img).getextrema()
            # 獲取圖像的平均像素值來更準確判斷是否需要反轉
            import numpy as np
            img_np = np.array(img)
            if img_np.mean() > 127: 
                # 白底黑字 -> 轉為 黑底白字 (因為 MNIST 是黑底白字)
                img = ImageOps.invert(img)

            # 顯示用的圖片 (放大一點比較好看)
            display_img = img.resize((200, 200))
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.canvas.create_image(100, 100, image=self.tk_img)

            # 預測
            self.predict(img)

        except Exception as e:
            messagebox.showerror("錯誤", f"無法讀取圖片：\n{e}")

    def predict(self, img):
        # 預處理
        img_tensor = self.transform(img) # 會自動 resize 到 28x28
        img_tensor = img_tensor.unsqueeze(0).to(self.device) # 增加 batch 維度

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_prob, pred_label = torch.max(probabilities, 1)

        digit = pred_label.item()
        prob = pred_prob.item() * 100

        self.lbl_result.config(text=f"預測結果：{digit}")
        self.lbl_prob.config(text=f"信心度：{prob:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()
