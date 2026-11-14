//! Test support: a receive that reports what its sending thread ended with.

use std::fmt::Debug;
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;

/// What `rx` receives next, with `sender` handed back. When the channel is closed instead,
/// panics with the error `sender` returned (or its own panic), which a bare
/// `recv().unwrap()` would hide behind `RecvError`.
pub(crate) fn sent<T, R, E: Debug>(
    rx: &Receiver<T>,
    sender: JoinHandle<Result<R, E>>,
) -> (T, JoinHandle<Result<R, E>>) {
    match rx.recv() {
        Ok(sent) => (sent, sender),
        Err(_) => match sender.join() {
            Ok(Err(e)) => panic!("the sending thread failed: {e:?}"),
            Ok(Ok(_)) => panic!("the sending thread ended without sending"),
            Err(p) => std::panic::resume_unwind(p),
        },
    }
}
