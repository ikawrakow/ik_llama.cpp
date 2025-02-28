/*
  hash_map -- Lock-Free Hash Map port from folly::AtomicUnorderedInsertMap for C++.

  Copyright (c) 2010-2017 <http://ez8.co> <orca.zhang@yahoo.com>

  This library is released under the MIT License.
  Please see LICENSE file or visit https://github.com/ez8-co/atomic for details.
 */
#pragma once

#include <stdexcept>
#include <memory>
#include <cstring>
#include <cassert>
#include <cstdio>

#ifdef _MSC_VER
  #include <intrin.h>
  #define LIKELY(x)                   (x)
  #define UNLIKELY(x)                 (x)
#else
  #define LIKELY(x)                   (__builtin_expect((x), 1))
  #define UNLIKELY(x)                 (__builtin_expect((x), 0))
#endif

#if __cplusplus >= 201103L || _MSC_VER >= 1700
  #include <atomic>
#else
namespace std {

  typedef enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
  } memory_order;

#ifdef _MSC_VER
  template <typename T, size_t N = sizeof(T)>
  struct interlocked {};

  template <typename T>
  struct interlocked<T, 4> {
    static inline T incre(T volatile* x) {
      return static_cast<T>(_InterlockedIncrement(reinterpret_cast<volatile long*>(x)));
    }
    static inline T decre(T volatile* x) {
      return static_cast<T>(_InterlockedDecrement(reinterpret_cast<volatile long*>(x)));
    }
    static inline T add(T volatile* x, T delta) {
      return static_cast<T>(_InterlockedExchangeAdd(reinterpret_cast<volatile long*>(x), delta));
    }
    static inline T compare_exchange(T volatile* x, const T new_val, const T expected_val) {
      return static_cast<T>(
        _InterlockedCompareExchange(reinterpret_cast<volatile long*>(x),
          static_cast<const long>(new_val), static_cast<const long>(expected_val)));
    }
    static inline T exchange(T volatile* x, const T new_val) {
      return static_cast<T>(
        _InterlockedExchange(
          reinterpret_cast<volatile long*>(x), static_cast<const long>(new_val)));
    }
  };

  template <typename T>
  struct interlocked<T, 8> {
    static inline T incre(T volatile* x) {
#ifdef WIN64
      return static_cast<T>(_InterlockedIncrement64(reinterpret_cast<volatile __int64*>(x)));
#else
      return add(x, 1);
#endif  // WIN64
    }
    static inline T decre(T volatile* x) {
#ifdef WIN64
      return static_cast<T>(_InterlockedDecrement64(reinterpret_cast<volatile __int64*>(x)));
#else
      return add(x, -1);
#endif  // WIN64
    }
    static inline T add(T volatile* x, T delta) {
#ifdef WIN64
      return static_cast<T>(_InterlockedExchangeAdd64(reinterpret_cast<volatile __int64*>(x), delta));
#else
      __int64 old_val, new_val;
      do {
        old_val = static_cast<__int64>(*x);
        new_val = old_val + static_cast<__int64>(delta);
      } while (_InterlockedCompareExchange64(
                 reinterpret_cast<volatile __int64*>(x), new_val, old_val) !=
               old_val);
      return static_cast<T>(new_val);
#endif  // WIN64
    }
    static inline T compare_exchange(T volatile* x, const T new_val, const T expected_val) {
      return static_cast<T>(
        _InterlockedCompareExchange64(reinterpret_cast<volatile __int64*>(x), 
          static_cast<const __int64>(new_val), static_cast<const __int64>(expected_val)));
    }
    static inline T exchange(T volatile* x, const T new_val) {
#ifdef WIN64
      return static_cast<T>(
        _InterlockedExchange64(reinterpret_cast<volatile __int64*>(x),
          static_cast<const __int64>(new_val)));
#else
      __int64 old_val;
      do {
        old_val = static_cast<__int64>(*x);
      } while (_InterlockedCompareExchange64(
                 reinterpret_cast<volatile __int64*>(x), new_val, old_val) !=
               old_val);
      return static_cast<T>(old_val);
#endif  // WIN64
    }
  };

#else

  template<typename>
  struct hash {};

  template<>
  struct hash<size_t> {
    inline size_t operator()(size_t v) const { return v; }
  };

#endif

  template <typename T>
  class atomic {
  public:
    atomic() : value_(static_cast<T>(0)) {}
    explicit atomic(const T value) : value_(value) {}

    T operator++() {
  #ifdef _MSC_VER
      return interlocked<T>::incre(&value_);
  #else
      return __atomic_add_fetch(&value_, 1, __ATOMIC_SEQ_CST);
  #endif
    }

    T operator++(int) {
      T v = load(); ++(*this); return v;
    }

    T operator--() {
  #ifdef _MSC_VER
      return interlocked<T>::decre(&value_);
  #else
      return __atomic_sub_fetch(&value_, 1, __ATOMIC_SEQ_CST);
  #endif
    }

    T operator+=(T v) {
  #ifdef _MSC_VER
      return interlocked<T>::add(&value_, v);
  #else
      return __atomic_add_fetch(&value_, v, __ATOMIC_SEQ_CST);
  #endif
    }

    bool compare_exchange_strong(T& expected_val, T new_val, memory_order order = memory_order_seq_cst) {
  #ifdef _MSC_VER
      return expected_val == interlocked<T>::compare_exchange(&value_, new_val, expected_val);
  #else
      return __atomic_compare_exchange_n(&value_, &expected_val, new_val, 0, order, __ATOMIC_SEQ_CST);
  #endif
    }

    void store(const T new_val, memory_order order = memory_order_seq_cst) {
  #ifdef _MSC_VER
      interlocked<T>::exchange(&value_, new_val);
  #else
      __atomic_store_n(&value_, new_val, order);
  #endif
    }

    T load(memory_order order = memory_order_seq_cst) const {
  #ifdef _MSC_VER
      return interlocked<T>::add(const_cast<volatile T*>(&value_), 0);
  #else
      return __atomic_load_n(&value_, order);
  #endif
    }

    T operator=(const T new_value) {
      store(new_value);
      return new_value;
    }

    operator T() const {
      return load();
    }

  private:
    volatile T value_;
  };
}
#endif

/*
* Copyright 2013-present Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
namespace lock_free {

  size_t nextPowTwo(size_t v) {
  #ifdef _MSC_VER
    unsigned long x = 0;
    _BitScanForward(&x, v - 1);
  #else
    int x = __builtin_clzll(v - 1);
  #endif
    return v ? (size_t(1) << (v - 1 ? (((sizeof(unsigned long long) << 3) - 1) ^ x) + 1 : 0)) : 1;
  }

  template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>,
    template <typename> class Atom = std::atomic,
    typename IndexType = size_t,
    typename Allocator = std::allocator<char> >

  struct hash_map {

  typedef Key key_type;
  typedef Value mapped_type;
  typedef std::pair<Key,Value> value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef Hash hasher;
  typedef KeyEqual key_equal;
  typedef const value_type& const_reference;

  typedef struct ConstIterator : public std::iterator<std::bidirectional_iterator_tag, value_type> {
    ConstIterator(const hash_map& owner, IndexType slot)
      : owner_(owner)
      , slot_(slot)
    {}

    const value_type& operator*() const {
      return owner_.slots_[slot_].keyValue();
    }

    const value_type* operator->() const {
      return &owner_.slots_[slot_].keyValue();
    }

    // pre-increment
    const ConstIterator& operator++() {
      while (slot_ > 0) {
        --slot_;
        if (owner_.slots_[slot_].state() == LINKED) {
          break;
        }
      }
      return *this;
    }

    // post-increment
    ConstIterator operator++(int /* dummy */) {
      ConstIterator prev = *this;
      ++*this;
      return prev;
    }

    bool operator==(const ConstIterator& rhs) const {
      return slot_ == rhs.slot_;
    }
    bool operator!=(const ConstIterator& rhs) const {
      return !(*this == rhs);
    }

  private:
    const hash_map& owner_;
    IndexType slot_;
  } const_iterator;

  friend ConstIterator;

  hash_map(size_t maxSize,
           float maxLoadFactor = 0.8f,
           const Allocator& alloc = Allocator())
    : allocator_(alloc)
  {
    size_t capacity = size_t(maxSize / (maxLoadFactor > 1.0f ? 1.0f : maxLoadFactor) + 128);
    size_t avail = size_t(1) << (8 * sizeof(IndexType) - 2);
    if (capacity > avail && maxSize < avail) {
      // we'll do our best
      capacity = avail;
    }
    if (capacity < maxSize || capacity > avail) {
      throw std::invalid_argument(
        "hash_map capacity must fit in IndexType with 2 bits "
        "left over");
    }

    numSlots_ = capacity;
    slotMask_ = nextPowTwo(capacity * 4) - 1;
    mmapRequested_ = sizeof(Slot) * capacity;
    slots_ = reinterpret_cast<Slot*>(allocator_.allocate(mmapRequested_));
    memset(slots_, 0, mmapRequested_);
    // mark the zero-th slot as in-use but not valid, since that happens
    // to be our nil value
    slots_[0].stateUpdate(EMPTY, CONSTRUCTING);
  }

  ~hash_map() {
    for (size_t i = 1; i < numSlots_; ++i) {
      slots_[i].~Slot();
    }
    allocator_.deallocate(reinterpret_cast<char*>(slots_), mmapRequested_);
  }

  template <typename Func, typename V>
  std::pair<const_iterator, bool> findOrConstruct(const Key& key, Func func, const V* value) {
    IndexType const slot = keyToSlotIdx(key);
    IndexType prev = slots_[slot].headAndState_.load(std::memory_order_acquire);

    IndexType existing = find(key, slot);
    if (existing)
      return std::make_pair(ConstIterator(*this, existing), false);

    IndexType idx = allocateNear(slot);
    // allocaion failed, return fake element
    if (!idx)
      return std::make_pair(ConstIterator(*this, idx), false);
    new (&slots_[idx].keyValue().first) Key(key);
    func(static_cast<void*>(&slots_[idx].keyValue().second), value);

    while (true) {
      slots_[idx].next_ = prev >> 2;

      // we can merge the head update and the CONSTRUCTING -> LINKED update
      // into a single CAS if slot == idx (which should happen often)
      IndexType after = idx << 2;
      if (slot == idx)
        after += LINKED;
      else
        after += (prev & 3);

      if (slots_[slot].headAndState_.compare_exchange_strong(prev, after)) {
        // success
        if (idx != slot)
          slots_[idx].stateUpdate(CONSTRUCTING, LINKED);
        return std::make_pair(ConstIterator(*this, idx), true);
      }
      // compare_exchange_strong updates its first arg on failure, so
      // there is no need to reread prev

      existing = find(key, slot);
      if (existing) {
        // our allocated key and value are no longer needed
        slots_[idx].keyValue().first.~Key();
        slots_[idx].keyValue().second.~Value();
        slots_[idx].stateUpdate(CONSTRUCTING, EMPTY);

        return std::make_pair(ConstIterator(*this, existing), false);
      }
    }
  }

  template <class K, class V>
  std::pair<const_iterator,bool> insert(const K& key, const V& value) {
    return findOrConstruct(key, &hash_map::copyCtor<V>, &value);
  }

  const_iterator find(const Key& key) const {
    return ConstIterator(*this, find(key, keyToSlotIdx(key)));
  }

  const_iterator cbegin() const {
    IndexType slot = numSlots_ - 1;
    while (slot > 0 && slots_[slot].state() != LINKED) {
      --slot;
    }
    return ConstIterator(*this, slot);
  }

  const_iterator cend() const {
    return ConstIterator(*this, 0);
  }

  // Add by orca.zhang@yahoo.com
  void clear() {
    for (size_t i = 1; i < numSlots_; ++i) {
      slots_[i].~Slot();
    }
    memset(slots_, 0, mmapRequested_);
    slots_[0].stateUpdate(EMPTY, CONSTRUCTING);
  }

  // Add by orca.zhang@yahoo.com
  bool erase(const Key& key) const {
    KeyEqual ke;
    IndexType slot = keyToSlotIdx(key);
    IndexType hs = slots_[slot].headAndState_.load(std::memory_order_acquire);
    IndexType last_slot = 0;
    for (IndexType idx = hs >> 2; idx != 0; idx = slots_[idx].next_) {
      if (ke(key, slots_[idx].keyValue().first)) {
        if (!last_slot)
          slots_[slot].headAndState_ = (slots_[idx].next_ & (unsigned)-4) | (hs & 3);
        else
          slots_[last_slot].next_ = slots_[idx].next_;
        slots_[idx].~Slot();
        slots_[idx].stateUpdate(LINKED, EMPTY);
        return true;
      }
      last_slot = idx;
    }
    return false;
  }

  private:
    enum {
      kMaxAllocationTries = 1000, // after this we throw
    };

    typedef IndexType BucketState;

    enum {
      EMPTY = 0,
      CONSTRUCTING = 1,
      LINKED = 2,
    };

    /// Lock-free insertion is easiest by prepending to collision chains.
    /// A large chaining hash table takes two cache misses instead of
    /// one, however.  Our solution is to colocate the bucket storage and
    /// the head storage, so that even though we are traversing chains we
    /// are likely to stay within the same cache line.  Just make sure to
    /// traverse head before looking at any keys.  This strategy gives us
    /// 32 bit pointers and fast iteration.
    struct Slot {
      /// The bottom two bits are the BucketState, the rest is the index
      /// of the first bucket for the chain whose keys map to this slot.
      /// When things are going well the head usually links to this slot,
      /// but that doesn't always have to happen.
      Atom<IndexType> headAndState_;

      /// The next bucket in the chain
      IndexType next_;

      /// Key and Value
      unsigned char raw_[sizeof(value_type)];

      ~Slot() {
        BucketState s = state();
        assert(s == EMPTY || s == LINKED);
        if (s == LINKED) {
          keyValue().first.~Key();
          keyValue().second.~Value();
        }
      }

      BucketState state() const {
        return BucketState(headAndState_.load(std::memory_order_acquire) & 3);
      }

      void stateUpdate(BucketState before, BucketState after) {
        assert(state() == before);
        headAndState_ += (after - before);
      }

      value_type& keyValue() {
        assert(state() != EMPTY);
        union {
          unsigned char* p;
          value_type* v;
        } u;
        u.p = raw_;
        return *u.v;
      }

      const value_type& keyValue() const {
        assert(state() != EMPTY);
        union {
          unsigned char* p;
          value_type* v;
        } u;
        u.p = raw_;
        return *u.v;
      }

    };

    // We manually manage the slot memory so we can bypass initialization
    // (by getting a zero-filled mmap chunk) and optionally destruction of
    // the slots

    size_t mmapRequested_;
    size_t numSlots_;

    /// tricky, see keyToSlodIdx
    size_t slotMask_;

    Allocator allocator_;
    Slot* slots_;

    IndexType keyToSlotIdx(const Key& key) const {
      size_t h = hasher()(key);
      h &= slotMask_;
      while (h >= numSlots_) {
        h -= numSlots_;
      }
      return h;
    }

    IndexType find(const Key& key, IndexType slot) const {
      KeyEqual ke;
      IndexType hs = slots_[slot].headAndState_.load(std::memory_order_acquire);
      for (slot = hs >> 2; slot != 0; slot = slots_[slot].next_) {
        if (ke(key, slots_[slot].keyValue().first)) {
          return slot;
        }
      }
      return 0;
    }

    /// Allocates a slot and returns its index.  Tries to put it near
    /// slots_[start].
    IndexType allocateNear(IndexType start) {
      for (IndexType tries = 0; tries < kMaxAllocationTries; ++tries) {
        IndexType slot = allocationAttempt(start, tries);
        IndexType prev = slots_[slot].headAndState_.load(std::memory_order_acquire);
        if ((prev & 3) == EMPTY &&
          slots_[slot].headAndState_.compare_exchange_strong(
            prev, prev + CONSTRUCTING - EMPTY)) {
          return slot;
        }
      }
      return 0; // return fake element rather than throw exception to ignore overflow
      // throw std::bad_alloc();
    }

    /// Returns the slot we should attempt to allocate after tries failed
    /// tries, starting from the specified slot.  This is pulled out so we
    /// can specialize it differently during deterministic testing
    IndexType allocationAttempt(IndexType start, IndexType tries) const {
      if (LIKELY(tries < 8 && start + tries < numSlots_)) {
        return IndexType(start + tries);
      } else {
        IndexType rv;
        if (sizeof(IndexType) <= 4) {
          rv = IndexType(rand() % numSlots_);
        } else {
          rv = IndexType(((int64_t(rand()) << 32) + rand()) % numSlots_);
        }
        assert(rv < numSlots_);
        return rv;
      }
    }

    template<typename V>
    static void copyCtor(void* raw, const V* v) {
      assert(v);
      new (raw) Value(*v);
    }
  };

  /// MutableAtom is a tiny wrapper than gives you the option of atomically
  /// updating values inserted into an hash_map<K,
  /// MutableAtom<V>>.  This relies on hash_map's guarantee
  /// that it doesn't move values.
  template <typename T, template <typename> class Atom = std::atomic>
  struct MutableAtom {
    mutable Atom<T> data;
    explicit MutableAtom(const T& init) : data(init) {}
  };

  /// MutableData is a tiny wrapper than gives you the option of using an
  /// external concurrency control mechanism to updating values inserted
  /// into an hash_map.
  template <typename T>
  struct MutableData {
    mutable T data;
    explicit MutableData(const T& init) : data(init) {}
  };

  /**
   * A very simple atomic single-linked list primitive.
   *
   * Usage:
   *
   * class MyClass {
   *   _linked_list_hook<MyClass> hook_;
   * }
   *
   * _linked_list<MyClass, &MyClass::hook_> list;
   * list.insert(&a);
   * list.sweep([] (MyClass* c) { doSomething(c); }
   */
  template <class T>
  struct _linked_list_hook {
    T* next{nullptr};
  };

  template <class T, _linked_list_hook<T> T::*HookMember>
  class _linked_list {
  public:
    _linked_list() {}

    _linked_list(const _linked_list&) = delete;
    _linked_list& operator=(const _linked_list&) =
        delete;

    _linked_list(_linked_list&& other) noexcept
        : head_(other.head_.exchange(nullptr, std::memory_order_acq_rel)) {}

    // Absent because would be too error-prone to use correctly because of
    // the requirement that lists are empty upon destruction.
    _linked_list& operator=(
        _linked_list&& other) noexcept = delete;

    /**
     * Move the currently held elements to a new list.
     * The current list becomes empty, but concurrent threads
     * might still add new elements to it.
     *
     * Equivalent to calling a move constructor, but more linter-friendly
     * in case you still need the old list.
     */
    _linked_list spliceAll() { return std::move(*this); }

    /**
     * Move-assign the current list to `other`, then reverse-sweep
     * the old list with the provided callback `func`.
     *
     * A safe replacement for the move assignment operator, which is absent
     * because of the resource leak concerns.
     */
    template <typename F>
    void reverseSweepAndAssign(_linked_list&& other, F&& func) {
      auto otherHead = other.head_.exchange(nullptr, std::memory_order_acq_rel);
      auto head = head_.exchange(otherHead, std::memory_order_acq_rel);
      unlinkAll(head, std::forward<F>(func));
    }

    /**
     * Note: The list must be empty on destruction.
     */
    ~_linked_list() { assert(empty()); }

    /**
     * Returns the current head of the list.
     *
     * WARNING: The returned pointer might not be valid if the list
     * is modified concurrently!
     */
    T* unsafeHead() const { return head_.load(std::memory_order_acquire); }

    /**
     * Returns true if the list is empty.
     *
     * WARNING: This method's return value is only valid for a snapshot
     * of the state, it might become stale as soon as it's returned.
     */
    bool empty() const { return unsafeHead() == nullptr; }

    /**
     * Atomically insert t at the head of the list.
     * @return True if the inserted element is the only one in the list
     *         after the call.
     */
    bool insertHead(T* t) {
      assert(next(t) == nullptr);

      auto oldHead = head_.load(std::memory_order_relaxed);
      do {
        next(t) = oldHead;
        /* oldHead is updated by the call below.

          NOTE: we don't use next(t) instead of oldHead directly due to
          compiler bugs (GCC prior to 4.8.3 (bug 60272), clang (bug 18899),
          MSVC (bug 819819); source:
          http://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange */
      } while (!head_.compare_exchange_weak(
          oldHead, t, std::memory_order_release, std::memory_order_relaxed));

      return oldHead == nullptr;
    }

    /**
     * Replaces the head with nullptr,
     * and calls func() on the removed elements in the order from tail to head.
     * Returns false if the list was empty.
     */
    template <typename F>
    bool sweepOnce(F&& func) {
      if (auto head = head_.exchange(nullptr, std::memory_order_acq_rel)) {
        auto rhead = reverse(head);
        unlinkAll(rhead, std::forward<F>(func));
        return true;
      }
      return false;
    }

    /**
     * Repeatedly replaces the head with nullptr,
     * and calls func() on the removed elements in the order from tail to head.
     * Stops when the list is empty.
     */
    template <typename F>
    void sweep(F&& func) {
      while (sweepOnce(func)) {
      }
    }

    /**
     * Similar to sweep() but calls func() on elements in LIFO order.
     *
     * func() is called for all elements in the list at the moment
     * reverseSweep() is called.  Unlike sweep() it does not loop to ensure the
     * list is empty at some point after the last invocation.  This way callers
     * can reason about the ordering: elements inserted since the last call to
     * reverseSweep() will be provided in LIFO order.
     *
     * Example: if elements are inserted in the order 1-2-3, the callback is
     * invoked 3-2-1.  If the callback moves elements onto a stack, popping off
     * the stack will produce the original insertion order 1-2-3.
     */
    template <typename F>
    void reverseSweep(F&& func) {
      // We don't loop like sweep() does because the overall order of callbacks
      // would be strand-wise LIFO which is meaningless to callers.
      auto head = head_.exchange(nullptr, std::memory_order_acq_rel);
      unlinkAll(head, std::forward<F>(func));
    }

  private:
    std::atomic<T*> head_{nullptr};

    static T*& next(T* t) { return (t->*HookMember).next; }

    /* Reverses a linked list, returning the pointer to the new head
      (old tail) */
    static T* reverse(T* head) {
      T* rhead = nullptr;
      while (head != nullptr) {
        auto t = head;
        head = next(t);
        next(t) = rhead;
        rhead = t;
      }
      return rhead;
    }

    /* Unlinks all elements in the linked list fragment pointed to by `head',
    * calling func() on every element */
    template <typename F>
    static void unlinkAll(T* head, F&& func) {
      while (head != nullptr) {
        auto t = head;
        head = next(t);
        next(t) = nullptr;
        func(t);
      }
    }
  };

  /**
   * A very simple atomic single-linked list primitive.
   *
   * Usage:
   *
   * linked_list<MyClass> list;
   * list.insert(a);
   * list.sweep([] (MyClass& c) { doSomething(c); }
   */

  template <class T>
  class linked_list {
  public:
    linked_list() {}
    linked_list(const linked_list&) = delete;
    linked_list& operator=(const linked_list&) = delete;
    linked_list(linked_list&& other) noexcept = default;
    linked_list& operator=(linked_list&& other) noexcept {
      list_.reverseSweepAndAssign(std::move(other.list_), [](Wrapper* node) {
        delete node;
      });
      return *this;
    }

    ~linked_list() {
      sweep([](T&&) {});
    }

    bool empty() const { return list_.empty(); }

    /**
     * Atomically insert t at the head of the list.
     * @return True if the inserted element is the only one in the list
     *         after the call.
     */
    bool insertHead(T t) {
      auto wrapper = std::make_unique<Wrapper>(std::move(t));

      return list_.insertHead(wrapper.release());
    }

    /**
     * Repeatedly pops element from head,
     * and calls func() on the removed elements in the order from tail to head.
     * Stops when the list is empty.
     */
    template <typename F>
    void sweep(F&& func) {
      list_.sweep([&](Wrapper* wrapperPtr) mutable {
        std::unique_ptr<Wrapper> wrapper(wrapperPtr);

        func(std::move(wrapper->data));
      });
    }

    /**
     * Sweeps the list a single time, as a single point in time swap with the
     * current contents of the list.
     *
     * Unlike sweep() it does not loop to ensure the list is empty at some point
     * after the last invocation.
     *
     * Returns false if the list is empty.
     */
    template <typename F>
    bool sweepOnce(F&& func) {
      return list_.sweepOnce([&](Wrapper* wrappedPtr) {
        std::unique_ptr<Wrapper> wrapper(wrappedPtr);
        func(std::move(wrapper->data));
      });
    }

    /**
     * Similar to sweep() but calls func() on elements in LIFO order.
     *
     * func() is called for all elements in the list at the moment
     * reverseSweep() is called.  Unlike sweep() it does not loop to ensure the
     * list is empty at some point after the last invocation.  This way callers
     * can reason about the ordering: elements inserted since the last call to
     * reverseSweep() will be provided in LIFO order.
     *
     * Example: if elements are inserted in the order 1-2-3, the callback is
     * invoked 3-2-1.  If the callback moves elements onto a stack, popping off
     * the stack will produce the original insertion order 1-2-3.
     */
    template <typename F>
    void reverseSweep(F&& func) {
      list_.reverseSweep([&](Wrapper* wrapperPtr) mutable {
        std::unique_ptr<Wrapper> wrapper(wrapperPtr);

        func(std::move(wrapper->data));
      });
    }

  private:
    struct Wrapper {
      explicit Wrapper(T&& t) : data(std::move(t)) {}

      _linked_list_hook<Wrapper> hook;
      T data;
    };
    _linked_list<Wrapper, &Wrapper::hook> list_;
  };

} // namespace lock_free