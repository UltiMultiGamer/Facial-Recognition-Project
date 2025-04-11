

# Facial Recognition Project  

uuuuuh facial recognition project for uni or sum 

made by meeeee :333333 [@UltiMultiGamer](https://github.com/UltiMultiGamer)  

---

##  How 2 Use  ???////????

### **Setup**  
1. make a venv:
   ```bash
   python -m venv venv
   ```
2. boot up the venv:

    ```bash
    .\venv\Scripts\activate
    ```
3. and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Boot**  
1. Start the server:  
   ```bash
   python face_server.py
   ```
2. Use `face_client.py` to send the server stuff:  
   ```bash
   python face_client.py [command]
   ```

---

## 📜 Commands 

| Command | Description | Example |
|---------|-------------|---------|
| `add` | Add a new face | `python face_client.py add --name "Debil"` |
| `recognize` | Recognize a face | `python face_client.py recognize` |
| `remove` | Remove a registered face | `python face_client.py remove --name "Debil"` |
| `list` | List all registered faces | `python face_client.py list` |

---
