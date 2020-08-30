use core::str::FromStr;

fn main() {
    println!("Hello, world!");
}

#[macro_use]
extern crate bitflags;

#[derive(PartialEq)]
#[derive(Debug)]
pub enum Error {
    ParseError,
    DecimalError(Signal, Option<String>, Decimal)
}

#[derive(PartialEq)]
#[derive(Debug)]
#[derive(Copy, Clone)]
pub enum Coefficient {
    Number(u64),
    QNaN,
    SNaN,
    Inf,
    NaN
}

#[derive(PartialEq)]
#[derive(Debug)]
#[derive(Copy, Clone)]
pub struct Decimal {
    sign: i8,
    coef: Coefficient,
    exp: i64
}

#[derive(PartialEq)]
#[derive(Debug)]
#[derive(Copy, Clone)]
pub enum Rounding {
    Down,
    HalfUp,
    HalfEven,
    Ceiling,
    Floor,
    HalfDown,
    Up
}

bitflags! {
    pub struct Signal: u8 {
        const NONE              = 0b00000000;
        const INVALID_OPERATION = 0b00000001;
        const DIVISION_BY_ZERO  = 0b00000010;
        const ROUNDED           = 0b00000100;
        const INEXACT           = 0b00001000;
    }
}

#[derive(Debug)]
pub struct Context {
    precision: u8,
    rounding: Rounding,
    flags: Signal,
    traps: Signal
}

pub fn default_context() -> Context {
    Context {
        precision: 28,
        rounding: Rounding::HalfUp,
        flags: Signal::NONE,
        traps: Signal::INVALID_OPERATION | Signal::DIVISION_BY_ZERO
    }
}

fn precision(num: Decimal, precision: u8, rounding: Rounding) -> (Decimal, Signal) {
    match num.coef {
        Coefficient::NaN => panic!("Unexpected NaN"),
        Coefficient::SNaN => (num, Signal::NONE),
        Coefficient::QNaN => (num, Signal::NONE),
        Coefficient::Inf => (num, Signal::NONE),
        Coefficient::Number(coef) => {
            let digits = coef.to_string();
            let num_digits = digits.chars().count();

            if num_digits > precision as usize {
                do_precision(num.sign, digits, num_digits, num.exp, precision as usize, rounding)
            } else {
                (num, Signal::NONE)
            }
        }
    }
}

fn do_precision(sign: i8, digits: String, num_digits: usize, mut exp: i64, mut precision: usize, rounding: Rounding) -> (Decimal, Signal)  {
    precision = std::cmp::min(num_digits, precision);
    let mut signif = digits.chars().take(precision).collect::<Vec<char>>();
    let remain = digits.chars().skip(precision).collect::<Vec<char>>();
    exp += remain.len() as i64;
    signif = if should_increment(rounding, sign, &signif, &remain) {
        digits_increment(signif)
    } else {
        signif
    };

    let signals = if any_nonzero(&remain) {
        Signal::INEXACT | Signal::ROUNDED
    } else {
        Signal::ROUNDED
    };

    
    let coef = digits_to_integer(signif);

    (Decimal{
        coef: Coefficient::Number(coef), 
        exp: exp,
        sign: sign
    }, signals)
}

fn should_increment(rounding: Rounding, sign: i8, signif: &Vec<char>, remain: &Vec<char>) -> bool {
    if remain.is_empty() {
        false
    } else {
        let mut iter = remain.iter().cloned();
        let first = iter.next().unwrap();
        let rest = iter.collect::<Vec<char>>();
        match rounding {
            Rounding::Down => false,
            Rounding::Up => true,
            Rounding::Ceiling => {
                sign == 1 && any_nonzero(remain)
            },
            Rounding::Floor => {
                sign == -1 && any_nonzero(remain)
            },
            Rounding::HalfUp => {
                first >= '5'
            },
            Rounding::HalfEven => {
                if first == '5' {
                    if signif.is_empty() {
                        any_nonzero(&rest)
                    } else {
                        any_nonzero(&rest) || last_odd(signif)
                    }
                } else {
                    first > '5'
                }
                
                
            },
            Rounding::HalfDown => {
                first > '5' || first == '5' && any_nonzero(&rest)
            }
        }
    }
}

fn last_odd(digits: &Vec<char>) -> bool {
    let last = digits.last().unwrap();
    let last_int = (*last as u8) - ('0' as u8);
    last_int % 2 == 1
}

fn any_nonzero(digits: &Vec<char>) -> bool {
    digits.iter().any(|&x| x != '0')
}

fn digits_increment(digits: Vec<char>) -> Vec<char> {
    let mut iter = digits.iter().cloned().rev();
    let mut v: Vec<char> = Vec::new();
    let mut inc = true;
    loop {
        match iter.next() {
            Some('9') => {
                if inc {
                    v.push('0')
                } else {
                    v.push('9')
                }
            },
            Some(c) => {
                if inc {
                    v.push(((c as u8) + 1) as char);
                    inc = false;
                } else {
                    v.push(c)
                }
            },
            None => {
                if inc {
                    v.push('1');
                }
                break
            }
        }
    }
    v.reverse();
    v
}

fn digits_to_integer(digits: Vec<char>) -> u64 {
    if digits.is_empty() {
        0
    } else {
        digits.iter().collect::<String>().parse::<u64>().unwrap()
    }
}





pub fn new(i: i64) -> Decimal {
    Decimal {
        sign: if i < 0 { -1 } else {1},
        coef: Coefficient::Number(i.abs() as u64),
        exp: 0
    }
}

pub fn is_nan(d: Decimal) -> bool {
    match d.coef {
        Coefficient::NaN => panic!("Unexpected NaN"),
        Coefficient::SNaN => true,
        Coefficient::QNaN => true,
        _ => false
    }
}

pub fn is_inf(d: Decimal) -> bool {
    match d.coef {
        Coefficient::NaN => panic!("Unexpected NaN"),
        Coefficient::Inf => true,
        _ => false
    }
}

fn pow10(n: u64) -> u64 {
    let mut acc = 1;
    for _i in 0..n {
        acc *= 10;
    }
    acc
}

fn add_align(coef1: u64, exp1: i64, coef2: u64, exp2: i64) -> (u64, u64) {
    if exp1 == exp2 {
        (coef1, coef2)
    } else if exp1 > exp2 {
        (coef1 * pow10((exp1 - exp2) as u64), coef2)
    } else {
        (coef1, coef2 * pow10((exp2 - exp1) as u64))
    }
}

fn add_sign(sign1: i8, sign2: i8, coef: i64, rounding: Rounding) -> i8 {
    if coef > 0 {
        1
    } else if coef < 0 {
        -1
    } else if sign1 == -1 && sign2 == -1 {
        -1
    } else if sign1 != sign2 && rounding == Rounding::Floor {
        -1
    } else {
        1
    }
}

impl Context {

    fn handle_error(&mut self, signal: Signal, reason: Option<String>, result: Decimal) -> Result<Decimal, Error> {
        self.flags = self.flags | signal;
    
        let error_signal = self.traps & signal;
    
        let nan = if error_signal != Signal::NONE {
            Coefficient::SNaN
        } else {
            Coefficient::QNaN
        };
    
        let modified_result = match result.coef {
            Coefficient::NaN => Decimal{coef: nan, sign: result.sign, exp: result.exp},
            _ => result
        };
    
        if error_signal != Signal::NONE {
            Err(Error::DecimalError(error_signal, reason, modified_result))
        } else {
            Ok(modified_result)
        }
    
    }

    fn context(&mut self, num: Decimal, signals: Signal) -> Result<Decimal, Error> {
        let (result, prec_signals) = precision(num, self.precision, self.rounding);
        self.handle_error(signals | prec_signals, Option::None, result)
    }

    pub fn abs(&mut self, d: Decimal) -> Result<Decimal, Error> {
        match d {
            Decimal{coef: Coefficient::NaN, sign: _, exp: _} => panic!("Unexpected NaN"),
            Decimal{coef: Coefficient::SNaN, sign: _, exp: _} => self.handle_error(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d),
            Decimal{coef: Coefficient::QNaN, sign: _, exp} => Ok(Decimal{coef: Coefficient::QNaN, sign: 1, exp: exp}),
            Decimal{coef: c, sign: _, exp: e} => self.context(Decimal{coef: c, sign: 1, exp: e}, Signal::NONE)
        }
    }

    pub fn minus(&mut self, d: Decimal) -> Result<Decimal, Error> {
        match d {
            Decimal{coef: Coefficient::NaN, sign: _, exp: _} => panic!("Unexpected NaN"),
            Decimal{coef: Coefficient::SNaN, sign: _, exp: _} => self.handle_error(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d),
            Decimal{coef: Coefficient::QNaN, sign: _, exp: _} => Ok(d),
            Decimal{coef: c, sign: s, exp: e} => self.context(Decimal{coef: c, sign: -s, exp: e}, Signal::NONE)
        }
    }

    pub fn plus(&mut self, d: Decimal) -> Result<Decimal, Error> {
        match d {
            Decimal{coef: Coefficient::NaN, sign: _, exp: _} => panic!("Unexpected NaN"),
            Decimal{coef: Coefficient::SNaN, sign: _, exp: _} => self.handle_error(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d),
            Decimal{coef: _, sign: _, exp: _} => self.context(d, Signal::NONE)
        }
    }

    pub fn add(&mut self, num1: Decimal, num2: Decimal) -> Result<Decimal, Error> {
        match (num1, num2) {
            (Decimal{coef: Coefficient::NaN, sign: _, exp: _}, _) => panic!("Unexpected NaN"),
            (_, Decimal{coef: Coefficient::NaN, sign: _, exp: _}) => panic!("Unexpected NaN"),
            (Decimal{coef: Coefficient::SNaN, sign: _, exp: _}, _) => self.handle_error(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), num1),
            (_, Decimal{coef: Coefficient::SNaN, sign: _, exp: _}) => self.handle_error(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), num2),
            (Decimal{coef: Coefficient::QNaN, sign: _, exp: _}, _) => Ok(num1),
            (_, Decimal{coef: Coefficient::QNaN, sign: _, exp: _}) => Ok(num2),
            (Decimal{coef: Coefficient::Inf, sign: sign1, exp: _}, Decimal{coef: Coefficient::Inf, sign: sign2, exp: _}) => {
                if sign1 == sign2 {
                    if num1.exp > num2.exp {
                        Ok(num1)
                    } else {
                        Ok(num2)
                    }
                } else {
                    self.handle_error(Signal::INVALID_OPERATION, Some("adding +Infinity and -Infinity".to_string()), Decimal{coef: Coefficient::NaN, sign: 1, exp: 0})
                }
            },
            (Decimal{coef: Coefficient::Inf, sign: _, exp: _}, _) => Ok(num1),
            (_, Decimal{coef: Coefficient::Inf, sign: _, exp: _}) => Ok(num2),
            (Decimal{coef: Coefficient::Number(coef1), sign: sign1, exp: exp1}, Decimal{coef: Coefficient::Number(coef2), sign: sign2, exp: exp2}) => {
                let (c1, c2) = add_align(coef1, exp1, coef2, exp2);
                let coef = sign1 as i64 * c1 as i64 + sign2 as i64 * c2 as i64;
                let exp = std::cmp::min(exp1, exp2);
                let sign = add_sign(sign1, sign2, coef, self.rounding);
                self.context(Decimal{sign: sign, coef: Coefficient::Number(coef.abs() as u64), exp: exp}, Signal::NONE)
            }
        }
    }

    pub fn sub(&mut self, num1: Decimal, num2: Decimal) -> Result<Decimal, Error> {
        self.add(num1, Decimal{sign: -num2.sign, ..num2})
    }
}

impl FromStr for Decimal {
    type Err = Error;

    fn from_str(s: &str) -> Result<Decimal, Error> {
        let downcased = s.to_lowercase();
        let mut iter = downcased.chars().peekable();
        match iter.peek() {
            Some('+') => {
                iter.next();
                parse_unsign(&mut iter)
            }
            Some('-') => {
                iter.next();
                match parse_unsign(&mut iter) {
                    Ok(Decimal{coef: c, sign: _, exp: e}) => Ok(Decimal{coef: c, sign: -1, exp: e}),
                    other => other
                }
            }
            _ => parse_unsign(&mut iter)
        }
    }

}

fn parse_unsign(iter: &mut std::iter::Peekable<std::str::Chars>) -> Result<Decimal, Error> {
    let int = parse_digits(iter);
    let float = parse_float(iter);
    let exp = parse_exp(iter);

    if iter.peek().is_some() || int.is_empty() && float.is_empty() {
        let collected = iter.collect::<String>();
        match collected.as_str() {
            "inf" | "infinity" => Ok(Decimal{coef: Coefficient::Inf, sign: 1, exp: 0}),
            "snan" => Ok(Decimal{coef: Coefficient::SNaN, sign: 1, exp: 0}),
            "nan" => Ok(Decimal{coef: Coefficient::QNaN, sign: 1, exp: 0}),
            _ => Err(Error::ParseError)
        }
    } else {
        let float_len = float.len() as i64;
        let int_float = [int, float].concat();
        let int_str: String = if int_float.is_empty() {String::from("0")} else {int_float.into_iter().collect()};
        let exp_str: String = if exp.is_empty() {String::from("0")} else {exp.into_iter().collect()};

        let coef_int = int_str.parse::<u64>().unwrap();
        let exp_int = exp_str.parse::<i64>().unwrap();
        
        Ok(Decimal{sign: 1, coef: Coefficient::Number(coef_int), exp: exp_int - float_len})
    }
}

fn parse_float(iter: &mut std::iter::Peekable<std::str::Chars>) -> Vec<char> {
    match iter.peek() {
        Some('.') => {
            iter.next();
            parse_digits(iter)
        }
        _ => Vec::new()
    }
}

fn parse_exp(iter: &mut std::iter::Peekable<std::str::Chars>) -> Vec<char> {
    match iter.peek() {
        Some('e') => {
            iter.next();
            match iter.peek() {
                Some(s) if is_sign(s) => {
                    let sign: char = *s;

                    iter.next();
                    let mut digits = parse_digits(iter);
                    
                    digits.insert(0, sign);
                    digits
                }

                _ => parse_digits(iter)
            }
        }
        _ => Vec::new()
    }
}

fn parse_digits(iter: &mut std::iter::Peekable<std::str::Chars>) -> Vec<char> {
    let mut v: Vec<char> = Vec::new();
    while let Some(c) = iter.peek() {
        if is_digit(c) {
            v.push(*c);
            iter.next();
        } else {
            break;
        }
    }
    v
}

fn is_digit(c: &char) -> bool {
    match c {
        '0'..='9' => true,
        _ => false
    }
}

fn is_sign(c: &char) -> bool {
    match c {
        '-' | '+' => true,
        _ => false
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn ds(s: &str) -> Decimal {
        do_parse(s).unwrap()
    }

    fn d(sign: i8, coef: Coefficient, exp: i64) -> Decimal {
        return Decimal {
            sign: sign,
            coef: coef,
            exp: exp
        };
    }

    fn do_parse(s: &str) -> Result<Decimal, Error> {
        s.parse::<Decimal>()
    }

    #[test]
    fn test_new_from_int() {
        let d = new(123);
        assert_eq!(d.sign, 1);
        assert_eq!(d.coef, Coefficient::Number(123));
        assert_eq!(d.exp, 0);

        let d = new(-123);
        assert_eq!(d.sign, -1);
        assert_eq!(d.coef, Coefficient::Number(123));
        assert_eq!(d.exp, 0);

        let d = new(0);
        assert_eq!(d.sign, 1);
        assert_eq!(d.coef, Coefficient::Number(0));
        assert_eq!(d.exp, 0);
    }

    #[test]
    fn test_new_parsing() {
        assert_eq!(Ok(d(1, Coefficient::Number(123), 0)), do_parse("123"));
        assert_eq!(Ok(d(1, Coefficient::Number(123), 0)), do_parse("+123"));
        assert_eq!(Ok(d(-1, Coefficient::Number(123), 0)), do_parse("-123"));

        assert_eq!(Ok(d(1, Coefficient::Number(1230), -1)), do_parse("123.0"));
        assert_eq!(Ok(d(1, Coefficient::Number(1230), -1)), do_parse("+123.0"));
        assert_eq!(Ok(d(-1, Coefficient::Number(1230), -1)), do_parse("-123.0"));

        assert_eq!(Ok(d(1, Coefficient::Number(15), -1)), do_parse("1.5"));
        assert_eq!(Ok(d(1, Coefficient::Number(15), -1)), do_parse("+1.5"));
        assert_eq!(Ok(d(-1, Coefficient::Number(15), -1)), do_parse("-1.5"));

        assert_eq!(Ok(d(1, Coefficient::Number(0), -1)), do_parse(".0"));
        assert_eq!(Ok(d(1, Coefficient::Number(0), 0)), do_parse("0."));

        assert_eq!(Ok(d(1, Coefficient::Number(0), 0)), do_parse("0"));
        assert_eq!(Ok(d(1, Coefficient::Number(0), 0)), do_parse("+0"));
        assert_eq!(Ok(d(-1, Coefficient::Number(0), 0)), do_parse("-0"));

        assert_eq!(Ok(d(1, Coefficient::Number(1230), 13)), do_parse("1230e13"));
        assert_eq!(Ok(d(1, Coefficient::Number(1230), 2)), do_parse("+1230e+2"));
        assert_eq!(Ok(d(-1, Coefficient::Number(1230), -2)), do_parse("-1230e-2"));
        
        assert_eq!(Ok(d(1, Coefficient::Number(123_000), 11)), do_parse("1230.00e13"));
        assert_eq!(Ok(d(1, Coefficient::Number(12_301_230), 1)), do_parse("+1230.1230e+5"));
        assert_eq!(Ok(d(-1, Coefficient::Number(123_001_010), -10)), do_parse("-1230.01010e-5"));

        assert_eq!(Ok(d(1, Coefficient::Inf, 0)), do_parse("inf"));
        assert_eq!(Ok(d(1, Coefficient::Inf, 0)), do_parse("infinity"));
        assert_eq!(Ok(d(-1, Coefficient::Inf, 0)), do_parse("-InfInitY"));

        assert_eq!(Ok(d(1, Coefficient::QNaN, 0)), do_parse("nAn"));
        assert_eq!(Ok(d(-1, Coefficient::QNaN, 0)), do_parse("-NaN"));

        assert_eq!(Ok(d(1, Coefficient::SNaN, 0)), do_parse("snAn"));
        assert_eq!(Ok(d(-1, Coefficient::SNaN, 0)), do_parse("-sNaN"));

        assert_eq!(Err(Error::ParseError), do_parse(""));
        assert_eq!(Err(Error::ParseError), do_parse("test"));
        assert_eq!(Err(Error::ParseError), do_parse("e0"));
        assert_eq!(Err(Error::ParseError), do_parse("42.+42"));
        assert_eq!(Err(Error::ParseError), do_parse("42e0.0"));
    }

    #[test]
    fn test_is_nan() {
        assert!(is_nan(d(1, Coefficient::SNaN, 1)));
        assert!(is_nan(d(1, Coefficient::QNaN, 1)));
        assert!(!is_nan(d(1, Coefficient::Inf, 1)));
        assert!(!is_nan(d(1, Coefficient::Number(1), 1)));
    }

    #[test]
    fn test_is_inf() {
        assert!(!is_inf(d(1, Coefficient::SNaN, 1)));
        assert!(!is_inf(d(1, Coefficient::QNaN, 1)));
        assert!(is_inf(d(1, Coefficient::Inf, 1)));
        assert!(!is_inf(d(1, Coefficient::Number(1), 1)));
    }

    #[test]
    fn test_plus() {
        let mut context = Context{precision: 2, ..default_context()};

        assert_eq!(context.plus(ds("0")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.plus(ds("5")).unwrap(), d(1, Coefficient::Number(5), 0));
        assert_eq!(context.plus(ds("123")).unwrap(), d(1, Coefficient::Number(12), 1));
        assert_eq!(context.plus(ds("nan")).unwrap(), d(1, Coefficient::QNaN, 0));

        assert_eq!(context.plus(ds("snan")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
    }

    #[test]
    fn test_minus() {
        let mut context = default_context();

        assert_eq!(context.minus(ds("0")).unwrap(), d(-1, Coefficient::Number(0), 0));
        assert_eq!(context.minus(ds("1")).unwrap(), d(-1, Coefficient::Number(1), 0));
        assert_eq!(context.minus(ds("-1")).unwrap(), d(1, Coefficient::Number(1), 0));
        assert_eq!(context.minus(ds("inf")).unwrap(), d(-1, Coefficient::Inf, 0));
        assert_eq!(context.minus(ds("nan")).unwrap(), d(1, Coefficient::QNaN, 0));

        assert_eq!(context.minus(ds("snan")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
    }

    #[test]
    fn test_abs() {
        let mut context = default_context();

        assert_eq!(context.abs(ds("123")).unwrap(), d(1, Coefficient::Number(123), 0));
        assert_eq!(context.abs(ds("-123")).unwrap(), d(1, Coefficient::Number(123), 0));
        assert_eq!(context.abs(ds("-12.5e2")).unwrap(), d(1, Coefficient::Number(125), 1));
        assert_eq!(context.abs(ds("-42e-42")).unwrap(), d(1, Coefficient::Number(42), -42));
        assert_eq!(context.abs(ds("-inf")).unwrap(), d(1, Coefficient::Inf, 0));
        assert_eq!(context.abs(ds("nan")).unwrap(), d(1, Coefficient::QNaN, 0));

        assert_eq!(context.abs(ds("snan")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
    }

    #[test]
    fn test_add() {
        let mut context = default_context();

        assert_eq!(context.add(ds("0"), ds("0")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.add(ds("1"), ds("1")).unwrap(), d(1, Coefficient::Number(2), 0));
        assert_eq!(context.add(ds("1.3e3"), ds("2.4e2")).unwrap(), d(1, Coefficient::Number(154), 1));
        assert_eq!(context.add(ds("0.42"), ds("-1.5")).unwrap(), d(-1, Coefficient::Number(108), -2));
        assert_eq!(context.add(ds("-2e-2"), ds("-2e-2")).unwrap(), d(-1, Coefficient::Number(4), -2));
        assert_eq!(context.add(ds("-0"), ds("0")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.add(ds("-0"), ds("-0")).unwrap(), d(-1, Coefficient::Number(0), 0));
        assert_eq!(context.add(ds("2"), ds("-2")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.add(ds("5"), ds("nan")).unwrap(), d(1, Coefficient::QNaN, 0));
        assert_eq!(context.add(ds("nan"), ds("5")).unwrap(), d(1, Coefficient::QNaN, 0));
        assert_eq!(context.add(ds("inf"), ds("inf")).unwrap(), d(1, Coefficient::Inf, 0));
        assert_eq!(context.add(ds("-inf"), ds("-inf")).unwrap(), d(-1, Coefficient::Inf, 0));
        assert_eq!(context.add(ds("inf"), ds("5")).unwrap(), d(1, Coefficient::Inf, 0));
        assert_eq!(context.add(ds("5"), ds("inf")).unwrap(), d(1, Coefficient::Inf, 0));

        assert_eq!(context.add(ds("-inf"), ds("inf")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("adding +Infinity and -Infinity".to_string()), d(1, Coefficient::SNaN, 0)));

        let mut context1 = Context{rounding: Rounding::Floor, ..default_context()};
        assert_eq!(context1.add(ds("2"), ds("-2")).unwrap(), d(-1, Coefficient::Number(0), 0));

        assert_eq!(context.add(ds("snan"), ds("0")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
        assert_eq!(context.add(ds("0"), ds("snan")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
    }

    #[test]
    fn test_sub() {
        let mut context = default_context();

        assert_eq!(context.sub(ds("0"), ds("0")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.sub(ds("1"), ds("2")).unwrap(), d(-1, Coefficient::Number(1), 0));
        assert_eq!(context.sub(ds("1.3e3"), ds("2.4e2")).unwrap(), d(1, Coefficient::Number(106), 1));
        assert_eq!(context.sub(ds("0.42"), ds("-1.5")).unwrap(), d(1, Coefficient::Number(192), -2));
        assert_eq!(context.sub(ds("2e-2"), ds("-2e-2")).unwrap(), d(1, Coefficient::Number(4), -2));
        assert_eq!(context.sub(ds("-0"), ds("0")).unwrap(), d(-1, Coefficient::Number(0), 0));
        assert_eq!(context.sub(ds("-0"), ds("-0")).unwrap(), d(1, Coefficient::Number(0), 0));
        assert_eq!(context.sub(ds("5"), ds("nan")).unwrap(), d(-1, Coefficient::QNaN, 0));
        assert_eq!(context.sub(ds("nan"), ds("5")).unwrap(), d(1, Coefficient::QNaN, 0));
        assert_eq!(context.sub(ds("inf"), ds("-inf")).unwrap(), d(1, Coefficient::Inf, 0));
        assert_eq!(context.sub(ds("-inf"), ds("inf")).unwrap(), d(-1, Coefficient::Inf, 0));
        assert_eq!(context.sub(ds("inf"), ds("5")).unwrap(), d(1, Coefficient::Inf, 0));
        assert_eq!(context.sub(ds("5"), ds("inf")).unwrap(), d(-1, Coefficient::Inf, 0));

        assert_eq!(context.sub(ds("inf"), ds("inf")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("adding +Infinity and -Infinity".to_string()), d(1, Coefficient::SNaN, 0)));

        let mut context1 = Context{rounding: Rounding::Floor, ..default_context()};
        assert_eq!(context1.sub(ds("2"), ds("2")).unwrap(), d(-1, Coefficient::Number(0), 0));

        assert_eq!(context.sub(ds("snan"), ds("0")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(1, Coefficient::SNaN, 0)));
        assert_eq!(context.sub(ds("0"), ds("snan")).unwrap_err(), Error::DecimalError(Signal::INVALID_OPERATION, Some("operation on NaN".to_string()), d(-1, Coefficient::SNaN, 0)));
    }
}
