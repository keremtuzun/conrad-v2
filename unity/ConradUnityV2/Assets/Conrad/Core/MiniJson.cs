// Minimal, allocation-tolerant JSON reader/writer for the Conrad wire protocol and config files.
// Objects -> Dictionary<string, object>, arrays -> List<object>, integers -> long, other numbers -> double.
// The writer emits canonical JSON (sorted keys, invariant culture) and refuses NaN/Infinity.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Conrad.UnityV2.Core
{
    public sealed class JsonException : Exception
    {
        public JsonException(string message) : base(message) { }
    }

    public static class MiniJson
    {
        public static object Parse(string text)
        {
            if (text == null) throw new JsonException("null JSON text");
            var p = new Parser(text);
            object value = p.ParseValue();
            p.SkipWhitespace();
            if (!p.AtEnd) throw new JsonException("trailing characters after JSON value at " + p.Position);
            return value;
        }

        public static Dictionary<string, object> ParseObject(string text)
        {
            var obj = Parse(text) as Dictionary<string, object>;
            if (obj == null) throw new JsonException("expected a JSON object");
            return obj;
        }

        public static string Serialize(object value)
        {
            var sb = new StringBuilder(256);
            Write(sb, value);
            return sb.ToString();
        }

        private static void Write(StringBuilder sb, object value)
        {
            switch (value)
            {
                case null: sb.Append("null"); return;
                case string s: WriteString(sb, s); return;
                case bool b: sb.Append(b ? "true" : "false"); return;
                case double d: WriteDouble(sb, d); return;
                case float f: WriteDouble(sb, f); return;
                case int i: sb.Append(i.ToString(CultureInfo.InvariantCulture)); return;
                case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); return;
                case IDictionary<string, object> dict:
                {
                    var keys = new List<string>(dict.Keys);
                    keys.Sort(string.CompareOrdinal);
                    sb.Append('{');
                    for (int k = 0; k < keys.Count; k++)
                    {
                        if (k > 0) sb.Append(',');
                        WriteString(sb, keys[k]);
                        sb.Append(':');
                        Write(sb, dict[keys[k]]);
                    }
                    sb.Append('}');
                    return;
                }
                case IEnumerable seq:
                {
                    sb.Append('[');
                    bool first = true;
                    foreach (object item in seq)
                    {
                        if (!first) sb.Append(',');
                        first = false;
                        Write(sb, item);
                    }
                    sb.Append(']');
                    return;
                }
                default:
                    throw new JsonException("cannot serialize " + value.GetType().Name);
            }
        }

        private static void WriteDouble(StringBuilder sb, double d)
        {
            if (double.IsNaN(d) || double.IsInfinity(d))
                throw new JsonException("non-finite number in JSON output");
            string s = d.ToString("R", CultureInfo.InvariantCulture);
            if (s.IndexOf('.') < 0 && s.IndexOf('E') < 0 && s.IndexOf('e') < 0) s += ".0";
            sb.Append(s);
        }

        private static void WriteString(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        private sealed class Parser
        {
            private readonly string _s;
            private int _i;

            public Parser(string s) { _s = s; }
            public bool AtEnd => _i >= _s.Length;
            public int Position => _i;

            public void SkipWhitespace()
            {
                while (_i < _s.Length && char.IsWhiteSpace(_s[_i])) _i++;
            }

            private char Peek()
            {
                SkipWhitespace();
                if (_i >= _s.Length) throw new JsonException("unexpected end of JSON");
                return _s[_i];
            }

            private void Expect(char c)
            {
                if (Peek() != c) throw new JsonException("expected '" + c + "' at " + _i);
                _i++;
            }

            public object ParseValue()
            {
                char c = Peek();
                switch (c)
                {
                    case '{': return ParseObjectBody();
                    case '[': return ParseArray();
                    case '"': return ParseString();
                    case 't': return Literal("true", true);
                    case 'f': return Literal("false", false);
                    case 'n': return Literal("null", null);
                    default: return ParseNumber();
                }
            }

            private object Literal(string word, object value)
            {
                if (string.CompareOrdinal(_s, _i, word, 0, word.Length) != 0)
                    throw new JsonException("invalid literal at " + _i);
                _i += word.Length;
                return value;
            }

            private Dictionary<string, object> ParseObjectBody()
            {
                var obj = new Dictionary<string, object>(StringComparer.Ordinal);
                Expect('{');
                if (Peek() == '}') { _i++; return obj; }
                while (true)
                {
                    string key = ParseString();
                    Expect(':');
                    if (obj.ContainsKey(key)) throw new JsonException("duplicate key '" + key + "'");
                    obj[key] = ParseValue();
                    char c = Peek();
                    _i++;
                    if (c == '}') return obj;
                    if (c != ',') throw new JsonException("expected ',' or '}' at " + (_i - 1));
                }
            }

            private List<object> ParseArray()
            {
                var list = new List<object>();
                Expect('[');
                if (Peek() == ']') { _i++; return list; }
                while (true)
                {
                    list.Add(ParseValue());
                    char c = Peek();
                    _i++;
                    if (c == ']') return list;
                    if (c != ',') throw new JsonException("expected ',' or ']' at " + (_i - 1));
                }
            }

            private string ParseString()
            {
                Expect('"');
                var sb = new StringBuilder();
                while (true)
                {
                    if (_i >= _s.Length) throw new JsonException("unterminated string");
                    char c = _s[_i++];
                    if (c == '"') return sb.ToString();
                    if (c != '\\') { sb.Append(c); continue; }
                    if (_i >= _s.Length) throw new JsonException("unterminated escape");
                    char e = _s[_i++];
                    switch (e)
                    {
                        case '"': sb.Append('"'); break;
                        case '\\': sb.Append('\\'); break;
                        case '/': sb.Append('/'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'n': sb.Append('\n'); break;
                        case 'r': sb.Append('\r'); break;
                        case 't': sb.Append('\t'); break;
                        case 'u':
                            if (_i + 4 > _s.Length) throw new JsonException("bad unicode escape");
                            sb.Append((char)int.Parse(_s.Substring(_i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                            _i += 4;
                            break;
                        default: throw new JsonException("bad escape \\" + e);
                    }
                }
            }

            private object ParseNumber()
            {
                int start = _i;
                bool isFloat = false;
                while (_i < _s.Length)
                {
                    char c = _s[_i];
                    if (c == '.' || c == 'e' || c == 'E') isFloat = true;
                    else if (!(char.IsDigit(c) || c == '-' || c == '+')) break;
                    _i++;
                }
                string token = _s.Substring(start, _i - start);
                if (token.Length == 0) throw new JsonException("unexpected character at " + start);
                if (!isFloat && long.TryParse(token, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l))
                    return l;
                if (double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
                    return d;
                throw new JsonException("invalid number '" + token + "'");
            }
        }
    }

    /// <summary>Typed accessors that fail loudly (a missing or mistyped field is a config error).</summary>
    public static class J
    {
        public static Dictionary<string, object> Obj(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v) || !(v is Dictionary<string, object> d))
                throw new JsonException("missing object '" + key + "'");
            return d;
        }

        public static List<object> Arr(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v) || !(v is List<object> a))
                throw new JsonException("missing array '" + key + "'");
            return a;
        }

        public static string Str(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v) || !(v is string s))
                throw new JsonException("missing string '" + key + "'");
            return s;
        }

        public static string StrOrNull(Dictionary<string, object> o, string key)
        {
            return o.TryGetValue(key, out object v) ? v as string : null;
        }

        public static bool Bool(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v) || !(v is bool b))
                throw new JsonException("missing bool '" + key + "'");
            return b;
        }

        public static double Num(object v, string what)
        {
            if (v is double d) return d;
            if (v is long l) return l;
            throw new JsonException("expected a number for '" + what + "'");
        }

        public static double Num(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v)) throw new JsonException("missing number '" + key + "'");
            return Num(v, key);
        }

        public static long Long(Dictionary<string, object> o, string key)
        {
            if (!o.TryGetValue(key, out object v) || !(v is long l))
                throw new JsonException("missing integer '" + key + "'");
            return l;
        }

        public static bool Has(Dictionary<string, object> o, string key)
        {
            return o.TryGetValue(key, out object v) && v != null;
        }
    }
}
