import torch
import torch.nn as nn
import torch.nn.functional as F
from trak import TRAKer
from trak.projectors import CudaProjector
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ============================================================================
# RANDOM SEED CONFIGURATION
# ============================================================================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For dataloader workers
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed}", flush=True)

# Call IMMEDIATELY after imports, before creating model
SEED = 42
set_seed(SEED)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet18(num_classes, channels=3):
    return ResNet(Block, [2, 2, 2, 2], num_classes, channels)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)


# ============================================================================
# DATA LOADING AND AUGMENTATION
# ============================================================================
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2470, 0.2435, 0.2616)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

def worker_init_fn(worker_id):
    """Seed workers for reproducibility"""
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

trainloader = torch.utils.data.DataLoader(
    train, 
    batch_size=128, 
    shuffle=True,
    num_workers=2,              # FIXED: Added num_workers
    worker_init_fn=worker_init_fn,
    pin_memory=True
)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    test, 
    batch_size=128, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================================
# MODEL SETUP
# ============================================================================

net = ResNet18(10).to('cuda')

# Weight initialization for better convergence
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

net.apply(init_weights)
print("✓ Model weights initialized with Kaiming initialization", flush=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(
    net.parameters(), 
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[60, 120, 160],
    gamma=0.2
)

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

EPOCHS = 200
START_EPOCH = 0
best_acc = 0.0
best_epoch = 0

print("="*80, flush=True)
print("RESNET18 TRAINING - PRODUCTION VERSION v4.0", flush=True)
print(f"Random Seed: {SEED}", flush=True)
print(f"Total Epochs: {EPOCHS}", flush=True)
print("="*80, flush=True)

# Check for existing checkpoints to resume training
checkpoint_files = sorted([f for f in os.listdir('checkpoints') if f.startswith('resnet_epoch_')])
if checkpoint_files:
    latest_file = checkpoint_files[-1]
    checkpoint = torch.load(os.path.join('checkpoints', latest_file))
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])      # FIXED: Load optimizer
    scheduler.load_state_dict(checkpoint['scheduler_state'])      # FIXED: Load scheduler
    START_EPOCH = checkpoint['epoch'] + 1
    if 'best_acc' in checkpoint:
        best_acc = checkpoint['best_acc']
    print(f"✓ Resuming from epoch {START_EPOCH} (loaded {latest_file})", flush=True)
    print(f"✓ Best accuracy so far: {best_acc:.2f}%", flush=True)
else:
    print("✓ Starting training from scratch", flush=True)

print(f"RUNNING IMAGE VERSION: EPOCHS = {EPOCHS} (starting from epoch {START_EPOCH})", flush=True)
print("="*80, flush=True)

# ============================================================================
# TRAINING LOOP
# ============================================================================

for epoch in range(START_EPOCH, EPOCHS):
    net.train()
    losses = []
    running_loss = 0
    correct_train = 0
    total_train = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        
        # Track training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if i % 100 == 0 and i > 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(trainloader)}], Loss: {running_loss/100:.4f}', flush=True)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    train_acc = 100 * correct_train / total_train
    scheduler.step()  # FIXED: Removed avg_loss argument
    
    # Evaluate on test set
    net.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_acc = 100 * correct_test / total_test
    avg_test_loss = test_loss / len(testloader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save checkpoint
    checkpoint_path = f"checkpoints/resnet_epoch_{epoch}.pt"
    torch.save({
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "avg_loss": avg_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "best_acc": best_acc
    }, checkpoint_path)
    
    # Track and save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        torch.save({
            "model_state": net.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc
        }, "checkpoints/best_model.pt")
        print(f"⭐ New best accuracy: {test_acc:.2f}% at epoch {epoch+1} (saved)", flush=True)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}", flush=True)
    
    # Clean up old checkpoints (keep last 5)
    checkpoint_files = sorted([f for f in os.listdir('checkpoints') if f.startswith('resnet_epoch_')])
    if len(checkpoint_files) > 5:
        for old_file in checkpoint_files[:-5]:
            os.remove(os.path.join('checkpoints', old_file))
    
    # Periodic memory cleanup
    if epoch % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()

print('Training Done', flush=True)
print(f"Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch+1}", flush=True)

# Load best model for TRAK analysis
if os.path.exists("checkpoints/best_model.pt"):
    checkpoint = torch.load("checkpoints/best_model.pt")
    net.load_state_dict(checkpoint['model_state'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1} with {checkpoint['test_acc']:.2f}% accuracy", flush=True)

# Final test accuracy verification
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f'Final Test Accuracy: {final_acc:.2f}%', flush=True)

# ============================================================================
# TRAK ATTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAK ATTRIBUTION ANALYSIS")
print("="*80 + "\n", flush=True)

# Clear memory before TRAK
del optimizer, scheduler
torch.cuda.empty_cache()
gc.collect()

# Verify loaded model
net.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Model accuracy for TRAK: {100*correct/total:.2f}%", flush=True)

# Get gradient dimension
grad_dim = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Gradient dimension: {grad_dim:,}")
print(f"Total parameters: {grad_dim/1e6:.2f}M", flush=True)

# Create non-shuffled trainloader for TRAK
trainloader_trak = torch.utils.data.DataLoader(
    train, 
    batch_size=128, 
    shuffle=False,
    num_workers=2,
    worker_init_fn=worker_init_fn
)
print("✓ Created trainloader_trak (non-shuffled)", flush=True)

# Initialize TRAKer
from trak.projectors import BasicProjector

traker = TRAKer(
    model=net,
    task='image_classification',
    train_set_size=len(train),
    save_dir='./trak_results',
    device='cuda',
    proj_dim=2048,
    projector=BasicProjector(
        grad_dim=grad_dim,
        proj_dim=2048,
        seed=SEED,
        proj_type='rademacher',
        device='cuda'
    )
)

print("TRAK initialized successfully", flush=True)

traker.load_checkpoint(checkpoint=net.state_dict(), model_id=0)
print("✅ Checkpoint loaded into TRAK - ready for featurization!", flush=True)

# Featurize training data
print("\nComputing gradient features for training data...", flush=True)

for batch_idx, (inputs, labels) in enumerate(trainloader_trak):
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    traker.featurize(batch=(inputs, labels), num_samples=inputs.size(0))

    if batch_idx % 100 == 0:
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Batch {batch_idx}/{len(trainloader_trak)} | GPU Memory: {gpu_mem:.2f}GB", flush=True)

    if batch_idx % 200 == 0 and batch_idx > 0:
        torch.cuda.empty_cache()
        gc.collect()

traker.finalize_features()
print("✅ Featurization complete!", flush=True)

# Get test samples
test_images, test_labels = next(iter(testloader))
test_images, test_labels = test_images.to('cuda'), test_labels.to('cuda')

num_test_samples = 10
test_subset = test_images[:num_test_samples]
test_labels_subset = test_labels[:num_test_samples]

print(f"\nSelected {num_test_samples} test samples for attribution analysis")
print(f"Test labels: {[classes[label.item()] for label in test_labels_subset]}", flush=True)

# Start scoring
traker.start_scoring_checkpoint(
    exp_name='test_attribution',
    checkpoint=net.state_dict(),
    model_id=0,
    num_targets=num_test_samples
)

print("\nComputing attribution scores...")
print("Phase 1: Scoring TRAINING data...", flush=True)



print("Phase 2: Scoring TEST samples...", flush=True)

# Score TEST samples ONLY
for i in range(num_test_samples):
    single_img = test_subset[i:i+1]
    single_label = test_labels_subset[i:i+1]
    
    # We feed the TEST image into .score()
    traker.score(batch=(single_img, single_label), num_samples=1)
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{num_test_samples} targets", flush=True)

# Finalize scores
print("\nFinalizing scores...", flush=True)
scores = traker.finalize_scores(exp_name='test_attribution')

# Transpose if needed
if scores.shape[0] == len(train):
    scores = scores.T
    print("Transposed scores for correct orientation", flush=True)

# Analyze top-k influential samples
k = 5

for test_idx in range(num_test_samples):
    test_label = test_labels_subset[test_idx].item()
    attribution_scores = scores[test_idx]
    top_k_indices = np.argsort(attribution_scores)[-k:][::-1]
    top_k_scores = attribution_scores[top_k_indices]

    print(f"\n{'='*60}")
    print(f"Test sample {test_idx} (True label: {classes[test_label]})")
    print(f"{'='*60}")
    print(f"Top {k} most influential training examples:")

    for rank, (train_idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        train_label = train[train_idx][1]
        match = "✓" if train_label == test_label else "✗"
        print(f"  {rank+1}. Train sample {train_idx:5d} | Label: {classes[train_label]:8s} {match} | Score: {score:7.4f}")

    if test_idx == 0:
        print(f"\nLeast influential (for comparison):")
        bottom_k_indices = np.argsort(attribution_scores)[:3]
        for rank, train_idx in enumerate(bottom_k_indices):
            train_label = train[train_idx][1]
            score = attribution_scores[train_idx]
            print(f"  {rank+1}. Train sample {train_idx:5d} | Label: {classes[train_label]:8s} | Score: {score:7.4f}")

# Visualize results
test_idx = 0
test_label = test_labels_subset[test_idx].item()
attribution_scores = scores[test_idx]
top_5_indices = np.argsort(attribution_scores)[-5:][::-1]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle(f'Test Sample (label: {classes[test_label]}) and Top-5 Influential Training Samples', fontsize=14)

def denormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = img.numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)

test_img = test_subset[test_idx].cpu()
axes[0, 0].imshow(denormalize(test_img))
axes[0, 0].set_title(f'TEST\n{classes[test_label]}', fontweight='bold', color='red')
axes[0, 0].axis('off')

axes[0, 1].axis('off')
axes[0, 2].axis('off')

for i, train_idx in enumerate(top_5_indices):
    train_img, train_label = train[train_idx]
    ax = axes[1, i] if i < 3 else axes[0, i-3+1]
    ax.imshow(denormalize(train_img))
    score = attribution_scores[train_idx]
    ax.set_title(f'#{i+1}: {classes[train_label]}\nScore: {score:.3f}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('top_influential_samples.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved to 'top_influential_samples.png'", flush=True)

np.save('attribution_scores.npy', scores)
print("✅ Attribution scores saved to 'attribution_scores.npy'", flush=True)

print("\n" + "="*80)
print("TRAK ANALYSIS COMPLETE!")
print(f"Best model accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
print("="*80, flush=True)
