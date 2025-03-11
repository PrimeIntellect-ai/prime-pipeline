use iroh::{
    endpoint::Connection as IrohConnection,
    protocol::{ProtocolHandler, Router},
    Endpoint, NodeAddr, PublicKey,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::wrap_pyfunction;
use pyo3::types::PyBytes;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

// ALPN protocol identifier
const ALPN: &[u8] = b"iroh-min/0";

/// Create an Iroh connector for peer-to-peer communication
#[pyfunction]
fn create_connector() -> PyResult<IrohConnector> {
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
            .accept(ALPN, ConnectionHandler::new(conn_clone))
            .spawn()
            .await
            .map_err(|e| format!("Failed to spawn router: {}", e))?;
            
        let node_addr = router.endpoint().node_addr().await
            .map_err(|e| format!("Failed to get node address: {}", e))?;
        
        let node_id = node_addr.node_id.to_string();
        
        Ok::<_, String>((endpoint, router, node_id))
    })
    .map_err(|e| PyRuntimeError::new_err(e))?;
    
    Ok(IrohConnector { 
        runtime,
        endpoint: Some(endpoint),
        router: Some(router),
        node_id,
        recv_connection,
        send_connection: Arc::new(Mutex::new(None)),
    })
}

/// Python class representing an Iroh connector
#[pyclass]
struct IrohConnector {
    runtime: Runtime,
    endpoint: Option<Endpoint>,
    router: Option<Router>,
    node_id: String,
    recv_connection: Arc<Mutex<Option<IrohConnection>>>,
    send_connection: Arc<Mutex<Option<IrohConnection>>>,
}

/// Handler for incoming connections
#[derive(Clone, Debug)]
struct ConnectionHandler {
    connection: Arc<Mutex<Option<IrohConnection>>>,
}

impl ConnectionHandler {
    fn new(connection: Arc<Mutex<Option<IrohConnection>>>) -> Self {
        Self { connection }
    }
}

impl ProtocolHandler for ConnectionHandler {
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

#[pymethods]
impl IrohConnector {
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
    fn send(&self, data: &[u8]) -> PyResult<()> {
        let lock = self.send_connection.lock().unwrap();
        let conn = match &*lock {
            Some(conn) => conn,
            None => return Err(PyRuntimeError::new_err("No connection established")),
        };
        
        self.runtime.block_on(async {
            let mut send = conn.open_uni().await?;
            send.write_all(data).await?;
            send.finish()?;
            send.stopped().await?;
            
            Ok::<_, anyhow::Error>(())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to send data: {}", e)))
    }
    
    /// Receive data from the connected peer
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        let lock = self.recv_connection.lock().unwrap();
        let conn = match &*lock {
            Some(conn) => conn,
            None => return Err(PyRuntimeError::new_err("No connection established")),
        };
        
        let data = self.runtime.block_on(async {
            let mut recv = conn.accept_uni().await?;
            let data = recv.read_to_end(usize::MAX).await?;
            Ok::<_, anyhow::Error>(data)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to receive data: {}", e)))?;
        
        Ok(PyBytes::new(py, &data).into())
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

/// A Python module implemented in Rust.
#[pymodule]
fn iroh_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_connector, m)?)?;
    m.add_class::<IrohConnector>()?;
    Ok(())
}
