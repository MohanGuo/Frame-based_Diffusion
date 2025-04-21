import torch

def get_rotation_matrix(theta):
    """
    Returns a 2D rotation matrix for angle theta (in radians)
    """
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ])

def test_rotation_equivariance():
    # Generate random 2D vectors
    X = torch.randn(2, 3)
    
    # Choose a random rotation angle
    theta = torch.rand(1) * 2 * torch.pi
    R = get_rotation_matrix(theta)
    
    # Method 1: First rotate, then QR decomposition
    RX = R @ X
    Q1, _ = torch.linalg.qr(RX)
    
    # Method 2: First QR decomposition, then rotate
    Q, _ = torch.linalg.qr(X)
    Q2 = R @ Q
    
    # Compare results (need to handle sign ambiguity)
    error = torch.min(
        torch.norm(Q1 - Q2),
        torch.norm(Q1 + Q2)  # QR decomposition is unique up to sign
    )
    
    print(f"Test vectors X:\n{X}")
    print(f"\nRotation angle: {theta.item():.2f} radians")
    print(f"\nMethod 1 (Rotate then QR):\n{Q1}")
    print(f"\nMethod 2 (QR then Rotate):\n{Q2}")
    print(f"\nError (Frobenius norm): {error:.2e}")
    
    return error < 1e-6

# Run multiple tests
torch.manual_seed(42)
num_tests = 5
all_passed = True

print("Testing rotation equivariance of QR decomposition...")
print("=" * 50)

for i in range(num_tests):
    print(f"\nTest {i+1}:")
    passed = test_rotation_equivariance()
    all_passed = all_passed and passed
    print(f"Test {i+1} {'passed' if passed else 'failed'}")
    print("-" * 50)

print(f"\nOverall result: {'All tests passed!' if all_passed else 'Some tests failed.'}")


def test_different_dimensions():
    dims = [(3, 2), (4, 3), (5, 4)]
    for dim in dims:
        print(f"\nTesting {dim[0]}x{dim[1]} matrix")
        X = torch.randn(*dim)
        Q1, _ = torch.linalg.qr(X)
        
        # Create a random rotation matrix in higher dimensions
        R = torch.linalg.qr(torch.randn(dim[0], dim[0]))[0]
        
        RX = R @ X
        Q2, _ = torch.linalg.qr(RX)
        
        # Verify R @ Q1 spans the same space as Q2
        error = torch.min(
            torch.norm(R @ Q1 - Q2),
            torch.norm(R @ Q1 + Q2)
        )
        print(f"Error: {error:.2e}")

print("\nTesting different dimensions:")
print("=" * 50)
test_different_dimensions()