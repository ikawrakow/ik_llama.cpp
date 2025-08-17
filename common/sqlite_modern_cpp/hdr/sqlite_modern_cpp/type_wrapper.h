#pragma once

#include <type_traits>
#include <string>
#include <memory>
#include <vector>
#ifdef __has_include
#if (__cplusplus >= 201703 || _MSVC_LANG >= 201703) && __has_include(<string_view>)
#define MODERN_SQLITE_STRINGVIEW_SUPPORT
#endif
#endif
#ifdef __has_include
#if (__cplusplus > 201402 || _MSVC_LANG > 201402) && __has_include(<optional>)
#define MODERN_SQLITE_STD_OPTIONAL_SUPPORT
#elif __has_include(<experimental/optional>) && __apple_build_version__ < 11000000
#define MODERN_SQLITE_EXPERIMENTAL_OPTIONAL_SUPPORT
#endif
#endif

#ifdef __has_include
#if (__cplusplus > 201402 || _MSVC_LANG > 201402) && __has_include(<variant>)
#define MODERN_SQLITE_STD_VARIANT_SUPPORT
#endif
#endif

#ifdef MODERN_SQLITE_STD_OPTIONAL_SUPPORT
#include <optional>
#endif

#ifdef MODERN_SQLITE_EXPERIMENTAL_OPTIONAL_SUPPORT
#include <experimental/optional>
#define MODERN_SQLITE_STD_OPTIONAL_SUPPORT
#endif

#ifdef MODERN_SQLITE_STD_VARIANT_SUPPORT
#include <variant>
#endif
#ifdef MODERN_SQLITE_STRINGVIEW_SUPPORT
#include <string_view>
namespace sqlite
{
	typedef const std::string_view str_ref;
	typedef const std::u16string_view u16str_ref;
}
#else
namespace sqlite
{
	typedef const std::string& str_ref;
	typedef const std::u16string& u16str_ref;
}
#endif
#include <sqlite3.h>
#include "errors.h"

namespace sqlite {
	template<class T, int Type, class = void>
	struct has_sqlite_type : std::false_type {};
	
	template<class T>
	using is_sqlite_value = std::integral_constant<bool, false
		|| has_sqlite_type<T, SQLITE_NULL>::value
		|| has_sqlite_type<T, SQLITE_INTEGER>::value
		|| has_sqlite_type<T, SQLITE_FLOAT>::value
		|| has_sqlite_type<T, SQLITE_TEXT>::value
		|| has_sqlite_type<T, SQLITE_BLOB>::value
	>;

	template<class T, int Type>
	struct has_sqlite_type<T&, Type> : has_sqlite_type<T, Type> {};
	template<class T, int Type>
	struct has_sqlite_type<const T, Type> : has_sqlite_type<T, Type> {};
	template<class T, int Type>
	struct has_sqlite_type<volatile T, Type> : has_sqlite_type<T, Type> {};

	template<class T>
	struct result_type {
		using type = T;
		constexpr result_type() = default;
		template<class U, class = typename std::enable_if<std::is_assignable<U, T>::value>>
		constexpr result_type(result_type<U>) { }
	};

	// int
	template<>
	struct has_sqlite_type<int, SQLITE_INTEGER> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const int& val) {
		return sqlite3_bind_int(stmt, inx, val);
	}
	inline void store_result_in_db(sqlite3_context* db, const int& val) {
		sqlite3_result_int(db, val);
	}
	inline int get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<int>) {
		return sqlite3_column_type(stmt, inx) == SQLITE_NULL ? 0 :
			sqlite3_column_int(stmt, inx);
	}
	inline int get_val_from_db(sqlite3_value *value, result_type<int>) {
		return sqlite3_value_type(value) == SQLITE_NULL ? 0 :
			sqlite3_value_int(value);
	}

	// sqlite_int64
	template<>
	struct has_sqlite_type<sqlite_int64, SQLITE_INTEGER, void> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const sqlite_int64& val) {
		return sqlite3_bind_int64(stmt, inx, val);
	}
	inline void store_result_in_db(sqlite3_context* db, const sqlite_int64& val) {
		sqlite3_result_int64(db, val);
	}
	inline sqlite_int64 get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<sqlite_int64 >) {
		return sqlite3_column_type(stmt, inx) == SQLITE_NULL ? 0 :
			sqlite3_column_int64(stmt, inx);
	}
	inline sqlite3_int64 get_val_from_db(sqlite3_value *value, result_type<sqlite3_int64>) {
		return sqlite3_value_type(value) == SQLITE_NULL ? 0 :
			sqlite3_value_int64(value);
	}

	// float
	template<>
	struct has_sqlite_type<float, SQLITE_FLOAT, void> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const float& val) {
		return sqlite3_bind_double(stmt, inx, double(val));
	}
	inline void store_result_in_db(sqlite3_context* db, const float& val) {
		sqlite3_result_double(db, val);
	}
	inline float get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<float>) {
		return sqlite3_column_type(stmt, inx) == SQLITE_NULL ? 0 :
			sqlite3_column_double(stmt, inx);
	}
	inline float get_val_from_db(sqlite3_value *value, result_type<float>) {
		return sqlite3_value_type(value) == SQLITE_NULL ? 0 :
			sqlite3_value_double(value);
	}

	// double
	template<>
	struct has_sqlite_type<double, SQLITE_FLOAT, void> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const double& val) {
		return sqlite3_bind_double(stmt, inx, val);
	}
	inline void store_result_in_db(sqlite3_context* db, const double& val) {
		sqlite3_result_double(db, val);
	}
	inline double get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<double>) {
		return sqlite3_column_type(stmt, inx) == SQLITE_NULL ? 0 :
			sqlite3_column_double(stmt, inx);
	}
	inline double get_val_from_db(sqlite3_value *value, result_type<double>) {
		return sqlite3_value_type(value) == SQLITE_NULL ? 0 :
			sqlite3_value_double(value);
	}

	/* for nullptr support */
	template<>
	struct has_sqlite_type<std::nullptr_t, SQLITE_NULL, void> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, std::nullptr_t) {
		return sqlite3_bind_null(stmt, inx);
	}
	inline void store_result_in_db(sqlite3_context* db, std::nullptr_t) {
		sqlite3_result_null(db);
	}

#ifdef MODERN_SQLITE_STD_VARIANT_SUPPORT
	template<>
	struct has_sqlite_type<std::monostate, SQLITE_NULL, void> : std::true_type {};

	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, std::monostate) {
		return sqlite3_bind_null(stmt, inx);
	}
	inline void store_result_in_db(sqlite3_context* db, std::monostate) {
		sqlite3_result_null(db);
	}
	inline std::monostate get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::monostate>) {
		return std::monostate();
	}
	inline std::monostate get_val_from_db(sqlite3_value *value, result_type<std::monostate>) {
		return std::monostate();
	}
#endif

	// str_ref
	template<>
	struct has_sqlite_type<std::string, SQLITE3_TEXT, void> : std::true_type {};
	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, str_ref val) {
		return sqlite3_bind_text(stmt, inx, val.data(), val.length(), SQLITE_TRANSIENT);
	}

	// Convert char* to string_view to trigger op<<(..., const str_ref )
	template<std::size_t N> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const char(&STR)[N]) {
		return sqlite3_bind_text(stmt, inx, &STR[0], N-1, SQLITE_TRANSIENT);
	}

	inline std::string get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::string>) {
		if ( sqlite3_column_type(stmt, inx) == SQLITE_NULL ) {
			return std::string();
		}
		char const * ptr = reinterpret_cast<char const *>(sqlite3_column_text(stmt, inx));
		// call sqlite3_column_text explicitely before sqlite3_column_bytes: it may convert the value to text
		return std::string(ptr, sqlite3_column_bytes(stmt, inx));
	}
	inline std::string  get_val_from_db(sqlite3_value *value, result_type<std::string>) {
		if ( sqlite3_value_type(value) == SQLITE_NULL ) {
			return std::string();
		}
		char const * ptr = reinterpret_cast<char const *>(sqlite3_value_text(value));
		// call sqlite3_column_text explicitely before sqlite3_column_bytes: it may convert the value to text
		return std::string(ptr, sqlite3_value_bytes(value));
	}

	inline void store_result_in_db(sqlite3_context* db, str_ref val) {
		sqlite3_result_text(db, val.data(), val.length(), SQLITE_TRANSIENT);
	}
	// u16str_ref
	template<>
	struct has_sqlite_type<std::u16string, SQLITE3_TEXT, void> : std::true_type {};
	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, u16str_ref val) {
		return sqlite3_bind_text16(stmt, inx, val.data(), sizeof(char16_t) * val.length(), SQLITE_TRANSIENT);
	}

	// Convert char* to string_view to trigger op<<(..., const str_ref )
	template<std::size_t N> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const char16_t(&STR)[N]) { 
		return sqlite3_bind_text16(stmt, inx, &STR[0], sizeof(char16_t) * (N-1), SQLITE_TRANSIENT);
	}

	inline std::u16string get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::u16string>) {
		if ( sqlite3_column_type(stmt, inx) == SQLITE_NULL ) {
			return std::u16string();
		}
		char16_t const * ptr = reinterpret_cast<char16_t const *>(sqlite3_column_text16(stmt, inx));
		// call sqlite3_column_text16 explicitely before sqlite3_column_bytes16: it may convert the value to text
		return std::u16string(ptr, sqlite3_column_bytes16(stmt, inx));
	}
	inline std::u16string  get_val_from_db(sqlite3_value *value, result_type<std::u16string>) {
		if ( sqlite3_value_type(value) == SQLITE_NULL ) {
			return std::u16string();
		}
		char16_t const * ptr = reinterpret_cast<char16_t const *>(sqlite3_value_text16(value));
		return std::u16string(ptr, sqlite3_value_bytes16(value));
	}

	inline void store_result_in_db(sqlite3_context* db, u16str_ref val) {
		sqlite3_result_text16(db, val.data(), sizeof(char16_t) * val.length(), SQLITE_TRANSIENT);
	}

	// Other integer types
	template<class Integral>
	struct has_sqlite_type<Integral, SQLITE_INTEGER, typename std::enable_if<std::is_integral<Integral>::value>::type> : std::true_type {};

	template<class Integral, class = typename std::enable_if<std::is_integral<Integral>::value>::type>
	inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const Integral& val) {
		return bind_col_in_db(stmt, inx, static_cast<sqlite3_int64>(val));
	}
	template<class Integral, class = std::enable_if<std::is_integral<Integral>::type>>
	inline void store_result_in_db(sqlite3_context* db, const Integral& val) {
		store_result_in_db(db, static_cast<sqlite3_int64>(val));
	}
	template<class Integral, class = typename std::enable_if<std::is_integral<Integral>::value>::type>
	inline Integral get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<Integral>) {
		return get_col_from_db(stmt, inx, result_type<sqlite3_int64>());
	}
	template<class Integral, class = typename std::enable_if<std::is_integral<Integral>::value>::type>
	inline Integral get_val_from_db(sqlite3_value *value, result_type<Integral>) {
		return get_val_from_db(value, result_type<sqlite3_int64>());
	}

	// vector<T, A>
	template<typename T, typename A>
	struct has_sqlite_type<std::vector<T, A>, SQLITE_BLOB, void> : std::true_type {};

	template<typename T, typename A> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const std::vector<T, A>& vec) {
		void const* buf = reinterpret_cast<void const *>(vec.data());
		int bytes = vec.size() * sizeof(T);
		return sqlite3_bind_blob(stmt, inx, buf, bytes, SQLITE_TRANSIENT);
	}
	template<typename T, typename A> inline void store_result_in_db(sqlite3_context* db, const std::vector<T, A>& vec) {
		void const* buf = reinterpret_cast<void const *>(vec.data());
		int bytes = vec.size() * sizeof(T);
		sqlite3_result_blob(db, buf, bytes, SQLITE_TRANSIENT);
	}
	template<typename T, typename A> inline std::vector<T, A> get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::vector<T, A>>) {
		if(sqlite3_column_type(stmt, inx) == SQLITE_NULL) {
			return {};
		}
		T const* buf = reinterpret_cast<T const *>(sqlite3_column_blob(stmt, inx));
		int bytes = sqlite3_column_bytes(stmt, inx);
		return std::vector<T, A>(buf, buf + bytes/sizeof(T));
	}
	template<typename T, typename A> inline std::vector<T, A> get_val_from_db(sqlite3_value *value, result_type<std::vector<T, A>>) {
		if(sqlite3_value_type(value) == SQLITE_NULL) {
			return {};
		}
		T const* buf = reinterpret_cast<T const *>(sqlite3_value_blob(value));
		int bytes = sqlite3_value_bytes(value);
		return std::vector<T, A>(buf, buf + bytes/sizeof(T));
	}

	/* for unique_ptr<T> support */
	template<typename T, int Type>
	struct has_sqlite_type<std::unique_ptr<T>, Type, void> : has_sqlite_type<T, Type> {};
	template<typename T>
	struct has_sqlite_type<std::unique_ptr<T>, SQLITE_NULL, void> : std::true_type {};

	template<typename T> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const std::unique_ptr<T>& val) {
		return val ? bind_col_in_db(stmt, inx, *val) : bind_col_in_db(stmt, inx, nullptr);
	}
	template<typename T> inline std::unique_ptr<T> get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::unique_ptr<T>>) {
		if(sqlite3_column_type(stmt, inx) == SQLITE_NULL) {
			return nullptr;
		}
		return std::make_unique<T>(get_col_from_db(stmt, inx, result_type<T>()));
	}
	template<typename T> inline std::unique_ptr<T> get_val_from_db(sqlite3_value *value, result_type<std::unique_ptr<T>>) {
		if(sqlite3_value_type(value) == SQLITE_NULL) {
			return nullptr;
		}
		return std::make_unique<T>(get_val_from_db(value, result_type<T>()));
	}

	// std::optional support for NULL values
#ifdef MODERN_SQLITE_STD_OPTIONAL_SUPPORT
#ifdef MODERN_SQLITE_EXPERIMENTAL_OPTIONAL_SUPPORT
	template<class T>
	using optional = std::experimental::optional<T>;
#else
	template<class T>
	using optional = std::optional<T>;
#endif

	template<typename T, int Type>
	struct has_sqlite_type<optional<T>, Type, void> : has_sqlite_type<T, Type> {};
	template<typename T>
	struct has_sqlite_type<optional<T>, SQLITE_NULL, void> : std::true_type {};

	template <typename OptionalT> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const optional<OptionalT>& val) {
		return val ? bind_col_in_db(stmt, inx, *val) : bind_col_in_db(stmt, inx, nullptr);
	}
	template <typename OptionalT> inline void store_result_in_db(sqlite3_context* db, const optional<OptionalT>& val) {
		if(val)
			store_result_in_db(db, *val);
		else
			sqlite3_result_null(db);
	}

	template <typename OptionalT> inline optional<OptionalT> get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<optional<OptionalT>>) {
		#ifdef MODERN_SQLITE_EXPERIMENTAL_OPTIONAL_SUPPORT
		if(sqlite3_column_type(stmt, inx) == SQLITE_NULL) {
			return std::experimental::nullopt;
		}
		return std::experimental::make_optional(get_col_from_db(stmt, inx, result_type<OptionalT>()));
		#else
		if(sqlite3_column_type(stmt, inx) == SQLITE_NULL) {
			return std::nullopt;
		}
		return std::make_optional(get_col_from_db(stmt, inx, result_type<OptionalT>()));
		#endif
	}
	template <typename OptionalT> inline optional<OptionalT> get_val_from_db(sqlite3_value *value, result_type<optional<OptionalT>>) {
		#ifdef MODERN_SQLITE_EXPERIMENTAL_OPTIONAL_SUPPORT
		if(sqlite3_value_type(value) == SQLITE_NULL) {
			return std::experimental::nullopt;
		}
		return std::experimental::make_optional(get_val_from_db(value, result_type<OptionalT>()));
		#else
		if(sqlite3_value_type(value) == SQLITE_NULL) {
			return std::nullopt;
		}
		return std::make_optional(get_val_from_db(value, result_type<OptionalT>()));
		#endif
	}
#endif

#ifdef MODERN_SQLITE_STD_VARIANT_SUPPORT
	namespace detail {
		template<class T, class U>
		struct tag_trait : U { using tag = T; };
	}

	template<int Type, class ...Options>
	struct has_sqlite_type<std::variant<Options...>, Type, void> : std::disjunction<detail::tag_trait<Options, has_sqlite_type<Options, Type>>...> {};

	namespace detail {
		template<int Type, typename ...Options, typename Callback, typename first_compatible = has_sqlite_type<std::variant<Options...>, Type>>
		inline std::variant<Options...> variant_select_type(Callback &&callback) {
			if constexpr(first_compatible::value)
				return callback(result_type<typename first_compatible::tag>());
			else
				throw errors::mismatch("The value is unsupported by this variant.", "", SQLITE_MISMATCH);
		}
		template<typename ...Options, typename Callback> inline decltype(auto) variant_select(int type, Callback &&callback) {
			switch(type) {
				case SQLITE_NULL:
					return variant_select_type<SQLITE_NULL, Options...>(std::forward<Callback>(callback));
				case SQLITE_INTEGER:
					return variant_select_type<SQLITE_INTEGER, Options...>(std::forward<Callback>(callback));
				case SQLITE_FLOAT:
					return variant_select_type<SQLITE_FLOAT, Options...>(std::forward<Callback>(callback));
				case SQLITE_TEXT:
					return variant_select_type<SQLITE_TEXT, Options...>(std::forward<Callback>(callback));
				case SQLITE_BLOB:
					return variant_select_type<SQLITE_BLOB, Options...>(std::forward<Callback>(callback));
			}
#ifdef _MSC_VER
			__assume(false);
#else
			__builtin_unreachable();
#endif
		}
	}
	template <typename ...Args> inline int bind_col_in_db(sqlite3_stmt* stmt, int inx, const std::variant<Args...>& val) {
		return std::visit([&](auto &&opt) {return bind_col_in_db(stmt, inx, std::forward<decltype(opt)>(opt));}, val);
	}
	template <typename ...Args> inline void store_result_in_db(sqlite3_context* db, const std::variant<Args...>& val) {
		std::visit([&](auto &&opt) {store_result_in_db(db, std::forward<decltype(opt)>(opt));}, val);
	}
	template <typename ...Args> inline std::variant<Args...> get_col_from_db(sqlite3_stmt* stmt, int inx, result_type<std::variant<Args...>>) {
		return detail::variant_select<Args...>(sqlite3_column_type(stmt, inx), [&](auto v) {
			return std::variant<Args...>(std::in_place_type<typename decltype(v)::type>, get_col_from_db(stmt, inx, v));
		});
	}
	template <typename ...Args> inline std::variant<Args...> get_val_from_db(sqlite3_value *value, result_type<std::variant<Args...>>) {
		return detail::variant_select<Args...>(sqlite3_value_type(value), [&](auto v) {
			return std::variant<Args...>(std::in_place_type<typename decltype(v)::type>, get_val_from_db(value, v));
		});
	}
#endif
}
