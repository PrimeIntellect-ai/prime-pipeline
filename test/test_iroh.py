from iroh_py import create_connector


def test_create_node():
    # Create a new node
    print("Creating connector...")
    node = create_connector()

    assert type(node.get_node_id()) is str, "Node ID should be a string"
    assert len(node.get_node_id()) == 64, "Node ID should be 64 characters long"
