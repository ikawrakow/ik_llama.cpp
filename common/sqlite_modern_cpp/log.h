#include "errors.h"

#include <sqlite3.h>

#include <utility>
#include <tuple>
#include <type_traits>

namespace sqlite {
	namespace detail {
		template<class>
		using void_t = void;
		template<class T, class = void>
		struct is_callable : std::false_type {};
		template<class Functor, class ...Arguments>
		struct is_callable<Functor(Arguments...), void_t<decltype(std::declval<Functor>()(std::declval<Arguments>()...))>> : std::true_type {};
		template<class Functor, class ...Functors>
		class FunctorOverload: public Functor, public FunctorOverload<Functors...> {
			public:
				template<class Functor1, class ...Remaining>
				FunctorOverload(Functor1 &&functor, Remaining &&... remaining):
					Functor(std::forward<Functor1>(functor)),
					FunctorOverload<Functors...>(std::forward<Remaining>(remaining)...) {}
				using Functor::operator();
				using FunctorOverload<Functors...>::operator();
		};
		template<class Functor>
		class FunctorOverload<Functor>: public Functor {
			public:
				template<class Functor1>
				FunctorOverload(Functor1 &&functor):
					Functor(std::forward<Functor1>(functor)) {}
				using Functor::operator();
		};
		template<class Functor>
		class WrapIntoFunctor: public Functor {
			public:
				template<class Functor1>
				WrapIntoFunctor(Functor1 &&functor):
					Functor(std::forward<Functor1>(functor)) {}
				using Functor::operator();
		};
		template<class ReturnType, class ...Arguments>
		class WrapIntoFunctor<ReturnType(*)(Arguments...)> {
			ReturnType(*ptr)(Arguments...);
			public:
				WrapIntoFunctor(ReturnType(*ptr)(Arguments...)): ptr(ptr) {}
				ReturnType operator()(Arguments... arguments) { return (*ptr)(std::forward<Arguments>(arguments)...); }
		};
		inline void store_error_log_data_pointer(std::shared_ptr<void> ptr) {
			static std::shared_ptr<void> stored;
			stored = std::move(ptr);
		}
		template<class T>
		std::shared_ptr<typename std::decay<T>::type> make_shared_inferred(T &&t) {
			return std::make_shared<typename std::decay<T>::type>(std::forward<T>(t));
		}
	}
	template<class Handler>
	typename std::enable_if<!detail::is_callable<Handler(const sqlite_exception&)>::value>::type
	error_log(Handler &&handler);
	template<class Handler>
	typename std::enable_if<detail::is_callable<Handler(const sqlite_exception&)>::value>::type
	error_log(Handler &&handler);
	template<class ...Handler>
	typename std::enable_if<sizeof...(Handler)>=2>::type
	error_log(Handler &&...handler) {
		return error_log(detail::FunctorOverload<detail::WrapIntoFunctor<typename std::decay<Handler>::type>...>(std::forward<Handler>(handler)...));
	}
	template<class Handler>
	typename std::enable_if<!detail::is_callable<Handler(const sqlite_exception&)>::value>::type
	error_log(Handler &&handler) {
		return error_log(std::forward<Handler>(handler), [](const sqlite_exception&) {});
	}
	template<class Handler>
	typename std::enable_if<detail::is_callable<Handler(const sqlite_exception&)>::value>::type
	error_log(Handler &&handler) {
		auto ptr = detail::make_shared_inferred([handler = std::forward<Handler>(handler)](int error_code, const char *errstr) mutable {
			  switch(error_code & 0xFF) {
#define SQLITE_MODERN_CPP_ERROR_CODE(NAME,name,derived)             \
				  case SQLITE_ ## NAME: switch(error_code) {                \
					  derived                                                 \
				    default: handler(errors::name(errstr, "", error_code)); \
				  };break;
#define SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED(BASE,SUB,base,sub)                    \
					  case SQLITE_ ## BASE ## _ ## SUB:                                       \
				      handler(errors::base ## _ ## sub(errstr, "", error_code)); \
				      break;
#include "lists/error_codes.h"
#undef SQLITE_MODERN_CPP_ERROR_CODE_EXTENDED
#undef SQLITE_MODERN_CPP_ERROR_CODE
				  default: handler(sqlite_exception(errstr, "", error_code)); \
			  }
			});

		sqlite3_config(SQLITE_CONFIG_LOG, static_cast<void(*)(void*,int,const char*)>([](void *functor, int error_code, const char *errstr) {
				(*static_cast<decltype(ptr.get())>(functor))(error_code, errstr);
			}), ptr.get());
		detail::store_error_log_data_pointer(std::move(ptr));
	}
}
