//! lib.rs
use iroh::{
    endpoint::Connection,
    protocol::{ProtocolHandler, Router},
    Endpoint, NodeAddr, PublicKey,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::wrap_pyfunction;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

// ALPN protocol identifier
const ALPN: &[u8] = b"iroh-py/0";

#[pyclass]
struct IrohNode {
    runtime: Runtime,
    endpoint: Option<Endpoint>,
    router: Option<Router>,
    node_id: String,
    recv_connection: Arc<Mutex<Option<Connection>>>,
    send_connection: Arc<Mutex<Option<Connection>>>,
}

/// Create an Iroh node for peer-to-peer communication
#[pyfunction]
fn create_node() -> PyResult<IrohNode> {
    // Create a new Tokio runtime
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;

    let recv_connection = Arc::new(Mutex::new(None));
    let conn_clone = Arc::clone(&recv_connection);
    
    let (endpoint, router, node_id) = runtime.block_on(async {
        let endpoint = Endpoint::builder().discovery_n0().bind().await
            .map_err(|e| format!("Failed to bind endpoint: {}", e))?;
        
        // Set up server functionality
        let router = Router::builder(endpoint.clone())
            .accept(ALPN, IrohReceiverHandler::new(conn_clone))
            .spawn()
            .await
            .map_err(|e| format!("Failed to spawn router: {}", e))?;

        let node_addr = router.endpoint().node_addr().await
            .map_err(|e| format!("Failed to get node address: {}", e))?;
        
        let node_id = node_addr.node_id.to_string();
        
        Ok::<_, String>((endpoint, router, node_id))
    })
    .map_err(|e| PyRuntimeError::new_err(e))?;

    
    Ok(IrohNode { 
        runtime,
        endpoint: Some(endpoint),
        router: Some(router),
        node_id,
        recv_connection,
        send_connection: Arc::new(Mutex::new(None)),
    })
}

#[pymethods]
impl IrohNode {
    /// Get the node ID
    fn get_node_id(&self) -> String {
        self.node_id.clone()
    }

    /// Connect to a remote node
    fn connect(&self, public_key_str: &str) -> PyResult<()> {
        let endpoint = match &self.endpoint {
            Some(ep) => ep,
            None => return Err(PyRuntimeError::new_err("Endpoint not initialized.")),
        };

        // Parse the public key
        let pk_bytes = hex::decode(public_key_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid hex string: {}", e)))?;
        
        if pk_bytes.len() != 32 {
            return Err(PyRuntimeError::new_err(format!("Expected a 32-byte public key, got {} bytes", pk_bytes.len())));
        }
        
        let mut pk_array = [0u8; 32];
        pk_array.copy_from_slice(&pk_bytes);
        
        let public_key = PublicKey::from_bytes(&pk_array)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid public key: {}", e)))?;
        
        let node_addr = NodeAddr::new(public_key);
        
        // Connect to the remote node
        let connection = self.runtime.block_on(async {
            endpoint.connect(node_addr, ALPN).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Connection failed: {}", e)))?;
        
        // Store the connection
        *self.send_connection.lock().unwrap() = Some(connection);
        
        Ok(())
    }
    
    /// Check if the node is ready to send and receive data
    fn is_ready(&self) -> bool {
        let can_send = self.send_connection.lock().unwrap().is_some();
        let can_recv = self.recv_connection.lock().unwrap().is_some();
        can_send && can_recv
    }
    
    /// Send data to the connected peer
    fn send<'p>(&self, py: Python<'p>, data: &[u8]) -> PyResult<Bound<'p, PyAny>> {
        let conn = {
            let lock = self.send_connection.lock().unwrap();
            match &*lock {
                Some(conn) => conn.clone(),
                None => return Err(PyRuntimeError::new_err("No connection established")),
            }
        };
        
        let data = data.to_vec();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut send = conn.open_uni().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.write_all(&data).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.finish().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.stopped().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }

    fn recv<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let conn = {
            let lock = self.recv_connection.lock().unwrap();
            match &*lock {
                Some(conn) => conn.clone(),
                None => return Err(PyRuntimeError::new_err("No connection established")),
            }
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut recv = conn.accept_uni().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let data = recv.read_to_end(usize::MAX).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(data)
        })
    }
    
    /// Shutdown the node
    fn shutdown(&self) -> PyResult<()> {
        if let Some(router) = &self.router {
            self.runtime.block_on(async {
                router.shutdown().await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to shutdown: {}", e)))?;
        }
        
        Ok(())
    }
}

/// Handler for incoming connections
#[derive(Clone, Debug)]
struct IrohReceiverHandler {
    connection: Arc<Mutex<Option<Connection>>>,
}

impl IrohReceiverHandler {
    fn new(connection: Arc<Mutex<Option<Connection>>>) -> Self {
        Self { connection }
    }
}

impl ProtocolHandler for IrohReceiverHandler {
    fn accept(&self, connecting: iroh::endpoint::Connecting) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>> {
        let connection = self.connection.clone();
        Box::pin(async move {
            let conn = connecting.await?;

            // Store only the first connection
            let mut lock = connection.lock().unwrap();
            if lock.is_none() {
                *lock = Some(conn);
            }
            
            Ok(())
        })
    }
}

/// Create an Iroh sender for outgoing peer-to-peer communication
#[pyfunction]
fn create_sender() -> PyResult<IrohSender> {
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
    
    let (endpoint, router) = runtime.block_on(async {
        let endpoint = Endpoint::builder().discovery_n0().bind().await
            .map_err(|e| format!("Failed to bind endpoint: {}", e))?;
        
        let router = Router::builder(endpoint.clone())
            .spawn()
            .await
            .map_err(|e| format!("Failed to spawn router: {}", e))?;
            
        Ok::<_, String>((endpoint, router))
    })
    .map_err(|e| PyRuntimeError::new_err(e))?;
    
    Ok(IrohSender { 
        runtime,
        endpoint: Some(endpoint),
        router: Some(router),
        connection: Arc::new(Mutex::new(None)),
    })
}

/// Create an Iroh receiver for incoming peer-to-peer communication
#[pyfunction]
fn create_receiver() -> PyResult<IrohReceiver> {
    // Create a new Tokio runtime
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;

    let recv_connection = Arc::new(Mutex::new(None));
    let conn_clone = Arc::clone(&recv_connection);
    
    let (endpoint, router, node_id) = runtime.block_on(async {
        let endpoint = Endpoint::builder().discovery_n0().bind().await
            .map_err(|e| format!("Failed to bind endpoint: {}", e))?;
        
        // Set up server functionality
        let router = Router::builder(endpoint.clone())
            .accept(ALPN, IrohReceiverHandler::new(conn_clone))
            .spawn()
            .await
            .map_err(|e| format!("Failed to spawn router: {}", e))?;
            
        let node_addr = router.endpoint().node_addr().await
            .map_err(|e| format!("Failed to get node address: {}", e))?;
        
        let node_id = node_addr.node_id.to_string();
        
        Ok::<_, String>((endpoint, router, node_id))
    })
    .map_err(|e| PyRuntimeError::new_err(e))?;
    
    Ok(IrohReceiver { 
        runtime,
        endpoint: Some(endpoint),
        router: Some(router),
        node_id,
        connection: recv_connection,
    })
}

/// Python class representing an Iroh sender
#[pyclass]
struct IrohSender {
    runtime: Runtime,
    endpoint: Option<Endpoint>,
    router: Option<Router>,
    connection: Arc<Mutex<Option<Connection>>>,
}

#[pymethods]
impl IrohSender {
    /// Connect to a remote node
    fn connect(&self, public_key_str: &str) -> PyResult<()> {
        let endpoint = match &self.endpoint {
            Some(ep) => ep,
            None => return Err(PyRuntimeError::new_err("Endpoint not initialized.")),
        };

        // Parse the public key
        let pk_bytes = hex::decode(public_key_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid hex string: {}", e)))?;
        
        if pk_bytes.len() != 32 {
            return Err(PyRuntimeError::new_err(format!("Expected a 32-byte public key, got {} bytes", pk_bytes.len())));
        }
        
        let mut pk_array = [0u8; 32];
        pk_array.copy_from_slice(&pk_bytes);
        
        let public_key = PublicKey::from_bytes(&pk_array)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid public key: {}", e)))?;
        
        let node_addr = NodeAddr::new(public_key);
        
        // Connect to the remote node
        let connection = self.runtime.block_on(async {
            endpoint.connect(node_addr, ALPN).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Connection failed: {}", e)))?;
        
        *self.connection.lock().unwrap() = Some(connection);
        
        Ok(())
    }

    /// Send data to the connected peer
    fn send<'p>(&self, py: Python<'p>, data: &[u8]) -> PyResult<Bound<'p, PyAny>> {
        let conn = {
            let lock = self.connection.lock().unwrap();
            match &*lock {
                Some(conn) => conn.clone(),
                None => return Err(PyRuntimeError::new_err("No connection established")),
            }
        };
        
        let data = data.to_vec();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut send = conn.open_uni().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.write_all(&data).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.finish().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            send.stopped().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }

    fn is_ready(&self) -> bool {
        let can_send = self.connection.lock().unwrap().is_some();
        can_send
    }

    /// Shutdown the sender
    fn shutdown(&self) -> PyResult<()> {
        if let Some(router) = &self.router {
            self.runtime.block_on(async {
                router.shutdown().await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to shutdown: {}", e)))?;
        }
        
        Ok(())
    }
}

/// Python class representing an Iroh receiver
#[pyclass]
struct IrohReceiver {
    runtime: Runtime,
    endpoint: Option<Endpoint>,
    router: Option<Router>,
    node_id: String,
    connection: Arc<Mutex<Option<Connection>>>,
}

#[pymethods]
impl IrohReceiver {
    /// Get the node ID
    fn get_node_id(&self) -> String {
        self.node_id.clone()
    }

    fn recv<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let conn = {
            let lock = self.connection.lock().unwrap();
            match &*lock {
                Some(conn) => conn.clone(),
                None => return Err(PyRuntimeError::new_err("No connection established")),
            }
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut recv = conn.accept_uni().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let data = recv.read_to_end(usize::MAX).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(data)
        })
    }

    fn is_ready(&self) -> bool {
        let can_recv = self.connection.lock().unwrap().is_some();
        can_recv
    }

    /// Shutdown the receiver
    fn shutdown(&self) -> PyResult<()> {
        if let Some(router) = &self.router {
            self.runtime.block_on(async {
                router.shutdown().await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to shutdown: {}", e)))?;
        }
        
        Ok(())
    }
}

#[pyfunction]
fn wait_for(py: Python, time: i64) -> PyResult<Bound<PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        tokio::time::sleep(std::time::Duration::from_secs(time as u64)).await;
        Ok(())
    })
}


#[pymodule]
fn iroh_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wait_for, m)?)?;
    m.add_function(wrap_pyfunction!(create_sender, m)?)?;
    m.add_function(wrap_pyfunction!(create_receiver, m)?)?;
    m.add_function(wrap_pyfunction!(create_node, m)?)?;
    m.add_class::<IrohSender>()?;
    m.add_class::<IrohReceiver>()?;
    m.add_class::<IrohNode>()?;
    Ok(())
}
