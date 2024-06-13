# Load the original model and set it to evaluation mode
original_model = models.resnet18(pretrained=False)
num_ftrs = original_model.fc.in_features
original_model.fc = nn.Linear(num_ftrs, 10)
original_model.load_state_dict(torch.load('best_model_state.pth'))
original_model.to(device)  # Move the original model to the specified device

# Load the test set
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def anamoly_detection_swap(model_wrapper, test_loader, device, n):
    model_wrapper.to(device)
    model_wrapper.eval()
    
    consecutive_errors = 0
    consecutive_corrects = 0
    using_quantized = True
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_wrapper(images)
            _, predicted = torch.max(outputs, 1)
            for idx in range(images.size(0)):
                if predicted[idx] == labels[idx]:
                    consecutive_errors = 0
                    consecutive_corrects += 1
                    correct += 1
                else:
                    consecutive_corrects = 0
                    consecutive_errors += 1

                # Check if we need to switch model precision
                if using_quantized and consecutive_errors > n:
                    model_wrapper.restore_full_precision()
                    using_quantized = False
                    print("Switched to full precision due to errors.")
                    consecutive_errors = 0  # Reset counter after switching
                elif not using_quantized and consecutive_corrects > n:
                    model_wrapper.quantize()
                    using_quantized = True
                    print("Switched back to quantized model due to correct classifications.")
                    consecutive_corrects = 0  # Reset counter after switching

                total += 1

    accuracy = correct / total
    print(f"Final accuracy: {accuracy:.4f}. Final swap lead to {'quantized' if using_quantized else 'full precision'} model")

bfp_model = BFPModelWrapper(original_model, group_size=8, mantissa_bits=9)
bfp_model.quantize()  # Start with the quantized model

anamoly_detection_swap(bfp_model, test_loader, device, n=3)
