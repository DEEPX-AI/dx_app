/**
 * @file safe_queue.hpp
 * @brief Thread-safe queue for async pipeline communication
 */

#ifndef DXAPP_SAFE_QUEUE_HPP
#define DXAPP_SAFE_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <queue>
#include <chrono>

namespace dxapp {

/**
 * @brief Thread-safe queue with bounded capacity
 * 
 * Provides blocking push/pop operations with optional timeout.
 * Used for passing data between pipeline stages in async processing.
 */
template <typename T>
class SafeQueue {
public:
    /**
     * @brief Construct a safe queue
     * @param max_size Maximum queue size (0 for unbounded)
     */
    explicit SafeQueue(size_t max_size = 100) : max_size_(max_size) {}

    /**
     * @brief Push an item to the queue (blocking if full)
     * @param item Item to push
     */
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        }
        queue_.push(std::move(item));
        condition_.notify_one();
    }

    /**
     * @brief Try to push an item with timeout
     * @param item Item to push
     * @param timeout_ms Timeout in milliseconds
     * @return true if item was pushed, false if timeout
     */
    bool tryPush(T item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            if (!condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                     [this] { return queue_.size() < max_size_; })) {
                return false;
            }
        }
        queue_.push(std::move(item));
        condition_.notify_one();
        return true;
    }

    /**
     * @brief Pop an item from the queue (blocking if empty)
     * @return Popped item
     */
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return item;
    }

    /**
     * @brief Try to pop an item with timeout
     * @param item Output item
     * @param timeout_ms Timeout in milliseconds
     * @return true if item was popped, false if timeout
     */
    bool tryPop(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !queue_.empty(); })) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return true;
    }

    /**
     * @brief Check if queue is empty
     * @return true if empty
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Get current queue size
     * @return Number of items in queue
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * @brief Clear all items from queue
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        condition_.notify_all();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;
};

}  // namespace dxapp

#endif  // DXAPP_SAFE_QUEUE_HPP
