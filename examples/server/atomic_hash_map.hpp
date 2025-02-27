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
namespace atomic {

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

} // namespace atomic