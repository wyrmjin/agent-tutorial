//! Async event stream based on `mpsc::unbounded_channel`.
//!
//! Provides push-based production + `Stream` trait consumption,
//! plus subscription callbacks for side-effect listeners.
//!
//! The [`EventStreamSender`] is the producer side (push events, mark done).
//! The [`EventStream`] is the consumer side (async iteration, subscriptions).

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

// ---------------------------------------------------------------------------
// Shared inner state (between sender and stream)
// ---------------------------------------------------------------------------

type ErasedListener<T> = Box<dyn Fn(&T) + Send + 'static>;

struct SharedInner<T> {
    listeners: std::sync::Mutex<Vec<(ErasedListener<T>, Arc<AtomicBool>)>>,
}

// ---------------------------------------------------------------------------
// Subscription guard — drop to unregister
// ---------------------------------------------------------------------------

/// RAII guard returned by [`EventStream::subscribe`].
/// Dropping it automatically unregisters the listener.
pub struct Subscription {
    alive: Arc<AtomicBool>,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// EventStreamSender — producer side
// ---------------------------------------------------------------------------

/// The producer side of an event stream.
///
/// Use [`push`] to send events and [`end`] to mark completion with a final result.
/// When this sender is dropped, the receiver side will eventually close.
pub struct EventStreamSender<T, R = ()> {
    tx: UnboundedSender<T>,
    result: Arc<std::sync::Mutex<Option<R>>>,
    done: Arc<AtomicBool>,
    shared: Arc<SharedInner<T>>,
}

impl<T: Send + 'static, R: Send + 'static> EventStreamSender<T, R> {
    /// Push an event. Called by the producer.
    ///
    /// Sends the event through the channel and notifies all registered
    /// subscription listeners synchronously.
    pub fn push(&self, event: T) {
        // Notify subscription listeners first (synchronous).
        let listeners = self.shared.listeners.lock().unwrap();
        for (listener, alive) in listeners.iter() {
            if alive.load(Ordering::Relaxed) {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (listener)(&event);
                }));
            }
        }
        drop(listeners);

        // Send through channel.
        let _ = self.tx.send(event);
    }

    /// Mark the stream as completed with a final result.
    pub fn end(&self, result: R) {
        *self.result.lock().unwrap() = Some(result);
        self.done.store(true, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// EventStream — consumer side
// ---------------------------------------------------------------------------

/// An async event stream with two consumption modes:
///
/// 1. **Stream iteration** — implements [`Stream`], use `while let Some(event) = stream.next().await`
/// 2. **Subscription callbacks** — register listeners via [`EventStream::subscribe`]
///
/// Type parameters:
/// - `T` — event type pushed by the producer
/// - `R` — final result type returned after the stream completes
pub struct EventStream<T, R = ()> {
    rx: Option<UnboundedReceiver<T>>,
    #[allow(dead_code)] // read via sender.end(), available for future result() API
    result: Arc<std::sync::Mutex<Option<R>>>,
    done: Arc<AtomicBool>,
    shared: Arc<SharedInner<T>>,
}

impl<T: Send + 'static, R: Send + 'static> EventStream<T, R> {
    /// Create a new event stream, returning the (sender, stream) pair.
    ///
    /// The sender is the producer side — use it to push events and mark completion.
    /// The stream is the consumer side — iterate it or subscribe listeners.
    pub fn new() -> (EventStreamSender<T, R>, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        let shared = Arc::new(SharedInner {
            listeners: std::sync::Mutex::new(Vec::new()),
        });
        let result = Arc::new(std::sync::Mutex::new(None));
        let done = Arc::new(AtomicBool::new(false));

        let sender = EventStreamSender {
            tx,
            result: Arc::clone(&result),
            done: Arc::clone(&done),
            shared: Arc::clone(&shared),
        };
        let stream = Self {
            rx: Some(rx),
            result,
            done,
            shared,
        };
        (sender, stream)
    }

    /// Register a subscription listener. Returns a [`Subscription`] guard.
    /// The listener is called synchronously for every event pushed.
    /// Drop the returned guard to unregister.
    pub fn subscribe(&self, listener: impl Fn(&T) + Send + 'static) -> Subscription {
        let alive = Arc::new(AtomicBool::new(true));
        let mut listeners = self.shared.listeners.lock().unwrap();
        listeners.push((Box::new(listener), Arc::clone(&alive)));
        Subscription { alive }
    }

    /// Check whether the stream has been marked as done.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Stream implementation
// ---------------------------------------------------------------------------

impl<T: Send + 'static, R: Send + 'static> Stream for EventStream<T, R> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let rx = match this.rx.as_mut() {
            Some(rx) => rx,
            None => return Poll::Ready(None),
        };
        let poll = Pin::new(rx).poll_recv(cx);
        // Once the receiver is closed, take it so subsequent polls return None.
        if let Poll::Ready(None) = &poll {
            this.rx.take();
        }
        poll
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_push_and_consume() {
        let (sender, stream) = EventStream::<i32, ()>::new();

        sender.push(1);
        sender.push(2);
        sender.push(3);
        drop(sender);

        let collected: Vec<i32> = stream.collect().await;
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_event_stream_push() {
        let (sender, stream) = EventStream::<String, ()>::new();

        sender.push("hello".to_string());
        sender.push("world".to_string());
        drop(sender);

        let collected: Vec<String> = stream.collect().await;
        assert_eq!(collected, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn test_subscription_receives_events() {
        let (sender, stream) = EventStream::<String, ()>::new();
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));

        let _sub = stream.subscribe({
            let received = Arc::clone(&received);
            move |event: &String| {
                received.lock().unwrap().push(event.clone());
            }
        });

        sender.push("hello".to_string());
        sender.push("world".to_string());

        tokio::task::yield_now().await;

        let items = received.lock().unwrap().clone();
        assert_eq!(items, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn test_subscription_unregister_on_drop() {
        let (sender, stream) = EventStream::<String, ()>::new();
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));

        {
            let _sub = stream.subscribe({
                let received = Arc::clone(&received);
                move |event: &String| {
                    received.lock().unwrap().push(event.clone());
                }
            });
            sender.push("first".to_string());
        } // _sub dropped here

        sender.push("second".to_string());
        tokio::task::yield_now().await;

        let items = received.lock().unwrap().clone();
        assert_eq!(items, vec!["first"]); // "second" not received
    }

    #[tokio::test]
    async fn test_end_with_result() {
        let (sender, stream) = EventStream::<i32, String>::new();
        sender.end("final".to_string());

        assert!(stream.is_done());
        let result = stream.result.lock().unwrap().clone();
        assert_eq!(result, Some("final".to_string()));
    }
}
