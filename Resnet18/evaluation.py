import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


baseline = pd.read_csv("model/baseline_history.csv")
cutout = pd.read_csv("model/cutout_history.csv")
cutmix = pd.read_csv("model/cutmix_history.csv")
mixup =  pd.read_csv("model/mixup_history.csv")

#show train loss
train_loss = pd.DataFrame()
train_loss["baseline"] = baseline["train_loss"]
train_loss["cutout"] = cutout["train_loss"]
train_loss["mixup"] = mixup["train_loss"]
train_loss["cutmix"] = cutmix["train_loss"]

plt.figure()
sns.lineplot(data = train_loss)
plt.xlabel("Epoch")
plt.title("Train Loss")
plt.ylabel("loss")
plt.legend()
plt.savefig("img/Train_Loss.png")
plt.close()


#show test accuracy
test_accuracy = pd.DataFrame()
test_accuracy["baseline"] = baseline["test_acc"]
test_accuracy["cutout"] = cutout["test_acc"]
test_accuracy["mixup"] = mixup["test_acc"]
test_accuracy["cutmix"] = cutmix["test_acc"]

plt.figure()
sns.lineplot(data = test_accuracy)
plt.xlabel("Epoch")
plt.title("Test Accuracy")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("img/Test_Accuracy.png")
plt.close()
