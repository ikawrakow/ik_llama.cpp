#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <functional>
#include <tuple>
#include <memory>

#define MODERN_SQLITE_VERSION 3002008

#include <sqlite3.h>

#include "sqlite_modern_cpp/type_wrapper.h"
#include "sqlite_modern_cpp/errors.h"
#include "sqlite_modern_cpp/utility/function_traits.h"
#include "sqlite_modern_cpp/utility/uncaught_exceptions.h"
#include "sqlite_modern_cpp/utility/utf16_utf8.h"

namespace sqlite {

	class database;
	class database_binder;

	template<std::size_t> class binder;

	typedef std::shared_ptr<sqlite3> connection_type;

	template<class T, bool Name = false>
	struct index_binding_helper {
		index_binding_helper(const index_binding_helper &) = delete;
#if __cplusplus < 201703 || _MSVC_LANG <= 201703
		index_binding_helper(index_binding_helper &&) = default;
#endif
		typename std::conditional<Name, const char *, int>::type index;
		T value;
	};

	template<class T>
	auto named_parameter(const char *name, T &&arg) {
		return index_binding_helper<decltype(arg), true>{name, std::forward<decltype(arg)>(arg)};
	}
	template<class T>
	auto indexed_parameter(int index, T &&arg) {
		return index_binding_helper<decltype(arg)>{index, std::forward<decltype(arg)>(arg)};
	}

	class row_iterator;
	class database_binder {

	public:
		// database_binder is not copyable
		database_binder() = delete;
		database_binder(const database_binder& other) = delete;
		database_binder& operator=(const database_binder&) = delete;

		database_binder(database_binder&& other) :
			_db(std::move(other._db)),
			_stmt(std::move(other._stmt)),
			_inx(other._inx), execution_started(other.execution_started) { }

		void execute();

		std::string sql() {
#if SQLITE_VERSION_NUMBER >= 3014000
			auto sqlite_deleter = [](void *ptr) {sqlite3_free(ptr);};
			std::unique_ptr<char, decltype(sqlite_deleter)> str(sqlite3_expanded_sql(_stmt.get()), sqlite_deleter);
			return str ? str.get() : original_sql();
#else
			return original_sql();
#endif
		}

		std::string original_sql() {
			return sqlite3_sql(_stmt.get());
		}

		void used(bool state) {
			if(!state) {
				// We may have to reset first if we haven't done so already:
				_next_index();
				--_inx;
			}
			execution_started = state;
		}
		bool used() const { return execution_started; }
		row_iterator begin();
		row_iterator end();

	private:
		std::shared_ptr<sqlite3> _db;
		std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> _stmt;
		utility::UncaughtExceptionDetector _has_uncaught_exception;

		int _inx;

		bool execution_started = false;

		int _next_index() {
			if(execution_started && !_inx) {
				sqlite3_reset(_stmt.get());
				sqlite3_clear_bindings(_stmt.get());
			}
			return ++_inx;
		}

		sqlite3_stmt* _prepare(u16str_ref sql) {
			return _prepare(utility::utf16_to_utf8(sql));
		}

		sqlite3_stmt* _prepare(str_ref sql) {
			int hresult;
			sqlite3_stmt* tmp = nullptr;
			const char *remaining;
			hresult = sqlite3_prepare_v2(_db.get(), sql.data(), sql.length(), &tmp, &remaining);
			if(hresult != SQLITE_OK) errors::throw_sqlite_error(hresult, sql, sqlite3_errmsg(_db.get()));
			if(!std::all_of(remaining, sql.data() + sql.size(), [](char ch) {return std::isspace(ch);}))
				throw errors::more_statements("Multiple semicolon separated statements are unsupported", sql);
			return tmp;
		}

		template<typename T> friend database_binder& operator<<(database_binder& db, T&&);
		template<typename T> friend database_binder& operator<<(database_binder& db, index_binding_helper<T>);
		template<typename T> friend database_binder& operator<<(database_binder& db, index_binding_helper<T, true>);
		friend void operator++(database_binder& db, int);

	public:

		database_binder(std::shared_ptr<sqlite3> db, u16str_ref sql):
			_db(db),
			_stmt(_prepare(sql), sqlite3_finalize),
			_inx(0) {
		}

		database_binder(std::shared_ptr<sqlite3> db, str_ref sql):
			_db(db),
			_stmt(_prepare(sql), sqlite3_finalize),
			_inx(0) {
		}

		~database_binder() noexcept(false) {
			/* Will be executed if no >>op is found, but not if an exception
			is in mid flight */
			if(!used() && !_has_uncaught_exception && _stmt) {
				execute();
			}
		}

		friend class row_iterator;
	};

	class row_iterator {
	public:
		class value_type {
		public:
			value_type(database_binder *_binder): _binder(_binder) {};
			template<class T>
			typename std::enable_if<is_sqlite_value<T>::value, value_type &>::type operator >>(T &result) {
				result = get_col_from_db(_binder->_stmt.get(), next_index++, result_type<T>());
				return *this;
			}
			template<class ...Types>
			value_type &operator >>(std::tuple<Types...>& values) {
				values = handle_tuple<std::tuple<typename std::decay<Types>::type...>>(std::index_sequence_for<Types...>());
				next_index += sizeof...(Types);
				return *this;
			}
			template<class ...Types>
			value_type &operator >>(std::tuple<Types...>&& values) {
				return *this >> values;
			}
			template<class ...Types>
			operator std::tuple<Types...>() {
				std::tuple<Types...> value;
				*this >> value;
				return value;
			}
			explicit operator bool() {
				return sqlite3_column_count(_binder->_stmt.get()) >= next_index;
			}
		private:
			template<class Tuple, std::size_t ...Index>
			Tuple handle_tuple(std::index_sequence<Index...>) {
				return Tuple(
						get_col_from_db(
							_binder->_stmt.get(),
							next_index + Index,
							result_type<typename std::tuple_element<Index, Tuple>::type>())...);
			}
			database_binder *_binder;
			int next_index = 0;
		};
		using difference_type = std::ptrdiff_t;
		using pointer = value_type*;
		using reference = value_type&;
		using iterator_category = std::input_iterator_tag;

		row_iterator() = default;
		explicit row_iterator(database_binder &binder): _binder(&binder) {
			_binder->_next_index();
			_binder->_inx = 0;
			_binder->used(true);
			++*this;
		}

		reference operator*() const { return value;}
		pointer operator->() const { return std::addressof(**this); }
		row_iterator &operator++() {
			switch(int result = sqlite3_step(_binder->_stmt.get())) {
				case SQLITE_ROW:
					value = {_binder};
					break;
				case SQLITE_DONE:
					_binder = nullptr;
					break;
				default:
					exceptions::throw_sqlite_error(result, _binder->sql(), sqlite3_errmsg(_binder->_db.get()));
			}
			return *this;
		}

		friend inline bool operator ==(const row_iterator &a, const row_iterator &b) {
			return a._binder == b._binder;
		}
		friend inline bool operator !=(const row_iterator &a, const row_iterator &b) {
			return !(a==b);
		}

	private:
		database_binder *_binder = nullptr;
		mutable value_type value{_binder}; // mutable, because `changing` the value is just reading it
	};

	inline row_iterator database_binder::begin() {
		return row_iterator(*this);
	}

	inline row_iterator database_binder::end() {
		return row_iterator();
	}

	namespace detail {
		template<class Callback>
		void _extract_single_value(database_binder &binder, Callback call_back) {
			auto iter = binder.begin();
			if(iter == binder.end())
				throw errors::no_rows("no rows to extract: exactly 1 row expected", binder.sql(), SQLITE_DONE);

			call_back(*iter);

			if(++iter != binder.end())
				throw errors::more_rows("not all rows extracted", binder.sql(), SQLITE_ROW);
		}
	}
	inline void database_binder::execute() {
		for(auto &&row : *this)
			(void)row;
	}
	namespace detail {
		template<class T> using void_t = void;
		template<class T, class = void>
		struct sqlite_direct_result : std::false_type {};
		template<class T>
		struct sqlite_direct_result<
			T,
			void_t<decltype(std::declval<row_iterator::value_type&>() >> std::declval<T&&>())>
		> : std::true_type {};
	}
	template <typename Result>
	inline typename std::enable_if<detail::sqlite_direct_result<Result>::value>::type operator>>(database_binder &binder, Result&& value) {
		detail::_extract_single_value(binder, [&value] (row_iterator::value_type &row) {
			row >> std::forward<Result>(value);
		});
	}

	template <typename Function>
	inline typename std::enable_if<!detail::sqlite_direct_result<Function>::value>::type operator>>(database_binder &db_binder, Function&& func) {
		using traits = utility::function_traits<Function>;

		for(auto &&row : db_binder) {
			binder<traits::arity>::run(row, func);
		}
	}

	template <typename Result>
	inline decltype(auto) operator>>(database_binder &&binder, Result&& value) {
		return binder >> std::forward<Result>(value);
	}

	namespace sql_function_binder {
		template<
			typename    ContextType,
			std::size_t Count,
			typename    Functions
		>
		inline void step(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals
		);

		template<
			std::size_t Count,
			typename    Functions,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) && sizeof...(Values) < Count), void>::type step(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals,
				Values&&...      values
		);

		template<
			std::size_t Count,
			typename    Functions,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) == Count), void>::type step(
				sqlite3_context* db,
				int,
				sqlite3_value**,
				Values&&...      values
		);

		template<
			typename    ContextType,
			typename    Functions
		>
		inline void final(sqlite3_context* db);

		template<
			std::size_t Count,
			typename    Function,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) < Count), void>::type scalar(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals,
				Values&&...      values
		);

		template<
			std::size_t Count,
			typename    Function,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) == Count), void>::type scalar(
				sqlite3_context* db,
				int,
				sqlite3_value**,
				Values&&...      values
		);
	}

	enum class OpenFlags {
		READONLY = SQLITE_OPEN_READONLY,
		READWRITE = SQLITE_OPEN_READWRITE,
		CREATE = SQLITE_OPEN_CREATE,
		NOMUTEX = SQLITE_OPEN_NOMUTEX,
		FULLMUTEX = SQLITE_OPEN_FULLMUTEX,
		SHAREDCACHE = SQLITE_OPEN_SHAREDCACHE,
		PRIVATECACH = SQLITE_OPEN_PRIVATECACHE,
		URI = SQLITE_OPEN_URI
	};
	inline OpenFlags operator|(const OpenFlags& a, const OpenFlags& b) {
		return static_cast<OpenFlags>(static_cast<int>(a) | static_cast<int>(b));
	}
	enum class Encoding {
		ANY = SQLITE_ANY,
		UTF8 = SQLITE_UTF8,
		UTF16 = SQLITE_UTF16
	};
	struct sqlite_config {
		OpenFlags flags = OpenFlags::READWRITE | OpenFlags::CREATE;
		const char *zVfs = nullptr;
		Encoding encoding = Encoding::ANY;
	};

	class database {
	protected:
		std::shared_ptr<sqlite3> _db;

	public:
		database(const std::string &db_name, const sqlite_config &config = {}): _db(nullptr) {
			sqlite3* tmp = nullptr;
			auto ret = sqlite3_open_v2(db_name.data(), &tmp, static_cast<int>(config.flags), config.zVfs);
			_db = std::shared_ptr<sqlite3>(tmp, [=](sqlite3* ptr) { sqlite3_close_v2(ptr); }); // this will close the connection eventually when no longer needed.
			if(ret != SQLITE_OK) errors::throw_sqlite_error(_db ? sqlite3_extended_errcode(_db.get()) : ret, {}, sqlite3_errmsg(_db.get()));
			sqlite3_extended_result_codes(_db.get(), true);
			if(config.encoding == Encoding::UTF16)
				*this << R"(PRAGMA encoding = "UTF-16";)";
		}

		database(const std::u16string &db_name, const sqlite_config &config = {}): database(utility::utf16_to_utf8(db_name), config) {
			if (config.encoding == Encoding::ANY)
				*this << R"(PRAGMA encoding = "UTF-16";)";
		}

		database(std::shared_ptr<sqlite3> db):
			_db(db) {}

		database_binder operator<<(str_ref sql) {
			return database_binder(_db, sql);
		}

		database_binder operator<<(u16str_ref sql) {
			return database_binder(_db, sql);
		}

		connection_type connection() const { return _db; }

		sqlite3_int64 last_insert_rowid() const {
			return sqlite3_last_insert_rowid(_db.get());
		}

		int rows_modified() const {
			return sqlite3_changes(_db.get());
		}

		template <typename Function>
		void define(const std::string &name, Function&& func) {
			typedef utility::function_traits<Function> traits;

			auto funcPtr = new auto(std::forward<Function>(func));
			if(int result = sqlite3_create_function_v2(
					_db.get(), name.data(), traits::arity, SQLITE_UTF8, funcPtr,
					sql_function_binder::scalar<traits::arity, typename std::remove_reference<Function>::type>,
					nullptr, nullptr, [](void* ptr){
				delete static_cast<decltype(funcPtr)>(ptr);
			}))
				errors::throw_sqlite_error(result, {}, sqlite3_errmsg(_db.get()));
		}

		template <typename StepFunction, typename FinalFunction>
		void define(const std::string &name, StepFunction&& step, FinalFunction&& final) {
			typedef utility::function_traits<StepFunction> traits;
			using ContextType = typename std::remove_reference<typename traits::template argument<0>>::type;

			auto funcPtr = new auto(std::make_pair(std::forward<StepFunction>(step), std::forward<FinalFunction>(final)));
			if(int result = sqlite3_create_function_v2(
					_db.get(), name.c_str(), traits::arity - 1, SQLITE_UTF8, funcPtr, nullptr,
					sql_function_binder::step<ContextType, traits::arity, typename std::remove_reference<decltype(*funcPtr)>::type>,
					sql_function_binder::final<ContextType, typename std::remove_reference<decltype(*funcPtr)>::type>,
					[](void* ptr){
				delete static_cast<decltype(funcPtr)>(ptr);
			}))
				errors::throw_sqlite_error(result, {}, sqlite3_errmsg(_db.get()));
		}

	};

	template<std::size_t Count>
	class binder {
	private:
		template <
			typename    Function,
			std::size_t Index
		>
			using nth_argument_type = typename utility::function_traits<
			Function
			>::template argument<Index>;

	public:
		// `Boundary` needs to be defaulted to `Count` so that the `run` function
		// template is not implicitly instantiated on class template instantiation.
		// Look up section 14.7.1 _Implicit instantiation_ of the ISO C++14 Standard
		// and the [dicussion](https://github.com/aminroosta/sqlite_modern_cpp/issues/8)
		// on Github.

		template<
			typename    Function,
			typename... Values,
			std::size_t Boundary = Count
		>
			static typename std::enable_if<(sizeof...(Values) < Boundary), void>::type run(
				row_iterator::value_type& row,
				Function&&                function,
				Values&&...               values
				) {
			typename std::decay<nth_argument_type<Function, sizeof...(Values)>>::type value;
			row >> value;
			run<Function>(row, function, std::forward<Values>(values)..., std::move(value));
		}

		template<
			typename    Function,
			typename... Values,
			std::size_t Boundary = Count
		>
			static typename std::enable_if<(sizeof...(Values) == Boundary), void>::type run(
				row_iterator::value_type&,
				Function&&                function,
				Values&&...               values
				) {
			function(std::move(values)...);
		}
	};

	// Some ppl are lazy so we have a operator for proper prep. statemant handling.
	void inline operator++(database_binder& db, int) { db.execute(); }

	template<typename T> database_binder &operator<<(database_binder& db, index_binding_helper<T> val) {
		db._next_index(); --db._inx;
		int result = bind_col_in_db(db._stmt.get(), val.index, std::forward<T>(val.value));
		if(result != SQLITE_OK)
			exceptions::throw_sqlite_error(result, db.sql(), sqlite3_errmsg(db._db.get()));
		return db;
	}

	template<typename T> database_binder &operator<<(database_binder& db, index_binding_helper<T, true> val) {
		db._next_index(); --db._inx;
		int index = sqlite3_bind_parameter_index(db._stmt.get(), val.index);
		if(!index)
			throw errors::unknown_binding("The given binding name is not valid for this statement", db.sql());
		int result = bind_col_in_db(db._stmt.get(), index, std::forward<T>(val.value));
		if(result != SQLITE_OK)
			exceptions::throw_sqlite_error(result, db.sql(), sqlite3_errmsg(db._db.get()));
		return db;
	}

	template<typename T> database_binder &operator<<(database_binder& db, T&& val) {
		int result = bind_col_in_db(db._stmt.get(), db._next_index(), std::forward<T>(val));
		if(result != SQLITE_OK)
			exceptions::throw_sqlite_error(result, db.sql(), sqlite3_errmsg(db._db.get()));
		return db;
	}
	// Convert the rValue binder to a reference and call first op<<, its needed for the call that creates the binder (be carefull of recursion here!)
	template<typename T> database_binder operator << (database_binder&& db, const T& val) { db << val; return std::move(db); }
	template<typename T, bool Name> database_binder operator << (database_binder&& db, index_binding_helper<T, Name> val) { db << index_binding_helper<T, Name>{val.index, std::forward<T>(val.value)}; return std::move(db); }

	namespace sql_function_binder {
		template<class T>
		struct AggregateCtxt {
			T obj;
			bool constructed = true;
		};

		template<
			typename ContextType,
			std::size_t Count,
			typename    Functions
		>
		inline void step(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals
		) {
			auto ctxt = static_cast<AggregateCtxt<ContextType>*>(sqlite3_aggregate_context(db, sizeof(AggregateCtxt<ContextType>)));
			if(!ctxt) return;
			try {
				if(!ctxt->constructed) new(ctxt) AggregateCtxt<ContextType>();
				step<Count, Functions>(db, count, vals, ctxt->obj);
				return;
			} catch(const sqlite_exception &e) {
				sqlite3_result_error_code(db, e.get_code());
				sqlite3_result_error(db, e.what(), -1);
			} catch(const std::exception &e) {
				sqlite3_result_error(db, e.what(), -1);
			} catch(...) {
				sqlite3_result_error(db, "Unknown error", -1);
			}
			if(ctxt && ctxt->constructed)
				ctxt->~AggregateCtxt();
		}

		template<
			std::size_t Count,
			typename    Functions,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) && sizeof...(Values) < Count), void>::type step(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals,
				Values&&...      values
		) {
			using arg_type = typename std::remove_cv<
					typename std::remove_reference<
							typename utility::function_traits<
									typename Functions::first_type
							>::template argument<sizeof...(Values)>
					>::type
			>::type;

			step<Count, Functions>(
					db,
					count,
					vals,
					std::forward<Values>(values)...,
					get_val_from_db(vals[sizeof...(Values) - 1], result_type<arg_type>()));
		}

		template<
			std::size_t Count,
			typename    Functions,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) == Count), void>::type step(
				sqlite3_context* db,
				int,
				sqlite3_value**,
				Values&&...      values
		) {
			static_cast<Functions*>(sqlite3_user_data(db))->first(std::forward<Values>(values)...);
		}

		template<
			typename    ContextType,
			typename    Functions
		>
		inline void final(sqlite3_context* db) {
			auto ctxt = static_cast<AggregateCtxt<ContextType>*>(sqlite3_aggregate_context(db, sizeof(AggregateCtxt<ContextType>)));
			try {
				if(!ctxt) return;
				if(!ctxt->constructed) new(ctxt) AggregateCtxt<ContextType>();
				store_result_in_db(db,
						static_cast<Functions*>(sqlite3_user_data(db))->second(ctxt->obj));
			} catch(const sqlite_exception &e) {
				sqlite3_result_error_code(db, e.get_code());
				sqlite3_result_error(db, e.what(), -1);
			} catch(const std::exception &e) {
				sqlite3_result_error(db, e.what(), -1);
			} catch(...) {
				sqlite3_result_error(db, "Unknown error", -1);
			}
			if(ctxt && ctxt->constructed)
				ctxt->~AggregateCtxt();
		}

		template<
			std::size_t Count,
			typename    Function,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) < Count), void>::type scalar(
				sqlite3_context* db,
				int              count,
				sqlite3_value**  vals,
				Values&&...      values
		) {
			using arg_type = typename std::remove_cv<
					typename std::remove_reference<
							typename utility::function_traits<Function>::template argument<sizeof...(Values)>
					>::type
			>::type;

			scalar<Count, Function>(
					db,
					count,
					vals,
					std::forward<Values>(values)...,
					get_val_from_db(vals[sizeof...(Values)], result_type<arg_type>()));
		}

		template<
			std::size_t Count,
			typename    Function,
			typename... Values
		>
		inline typename std::enable_if<(sizeof...(Values) == Count), void>::type scalar(
				sqlite3_context* db,
				int,
				sqlite3_value**,
				Values&&...      values
		) {
			try {
				store_result_in_db(db,
						(*static_cast<Function*>(sqlite3_user_data(db)))(std::forward<Values>(values)...));
			} catch(const sqlite_exception &e) {
				sqlite3_result_error_code(db, e.get_code());
				sqlite3_result_error(db, e.what(), -1);
			} catch(const std::exception &e) {
				sqlite3_result_error(db, e.what(), -1);
			} catch(...) {
				sqlite3_result_error(db, "Unknown error", -1);
			}
		}
	}
}

