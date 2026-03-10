"""
report_generator.py
將分析結果輸出為美觀的 HTML 報告
每張卡片內嵌迷你 K 線 + MACD + RSI 圖
"""

import json
from datetime import datetime


ACTION_CONFIG = {
    "加碼": {"color": "#3fb950", "bg": "rgba(63,185,80,0.12)", "icon": "▲"},
    "持有": {"color": "#58a6ff", "bg": "rgba(88,166,255,0.12)", "icon": "◆"},
    "觀望": {"color": "#e3b341", "bg": "rgba(227,179,65,0.12)", "icon": "◉"},
    "減碼": {"color": "#ffa657", "bg": "rgba(255,166,87,0.12)", "icon": "▽"},
    "停損": {"color": "#f85149", "bg": "rgba(248,81,73,0.12)", "icon": "✕"},
}


def _pct_color(pct):
    if pct > 0:   return "#f85149"
    elif pct < 0: return "#3fb950"
    return "#8b949e"


def _pct_str(pct):
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.2f}%"


def _profit_color(pct):
    if pct > 5:    return "#f85149"
    elif pct > 0:  return "#ffa657"
    elif pct < -5: return "#3fb950"
    elif pct < 0:  return "#58a6ff"
    return "#8b949e"


def generate_stock_card(stock_data):
    stock_id   = stock_data["stock_id"]
    name       = stock_data["name"]
    portfolio  = stock_data["portfolio"]
    signals    = stock_data["signals"]
    analysis   = stock_data["analysis"]
    chart_data = stock_data.get("chart_data", {})

    fundamentals = stock_data.get("fundamentals", {})
    cost        = portfolio.get("cost_price", 0)
    shares      = portfolio.get("shares", 0)
    shares_str  = str(int(shares)) if shares == int(shares) else str(shares)
    close       = signals.get("close", 0)
    change      = signals.get("change", 0)
    change_pct  = signals.get("change_pct", 0)
    vol_ratio   = signals.get("vol_ratio", 1) or 1
    rsi         = signals.get("rsi", "-")
    macd_hist   = signals.get("macd_hist", 0) or 0
    macd_status = signals.get("macd_status", "-")
    rsi_status  = signals.get("rsi_status", "-")

    # 營收摘要
    rev = fundamentals.get("revenue", {})
    rev_yoy = rev.get("年增率(%)")
    rev_mom = rev.get("月增率(%)")
    # 籌碼摘要（外資 + 投信今日）
    inst = fundamentals.get("institutional", {})
    foreign_net = inst.get("外資_今日(張)")
    trust_net   = inst.get("投信_今日(張)")

    profit_pct    = ((close - cost) / cost * 100) if cost else 0
    profit_amount = (close - cost) * shares if cost else 0

    action     = analysis.get("action", "觀望")
    action_cfg = ACTION_CONFIG.get(action, ACTION_CONFIG["觀望"])

    safe_id = stock_id.replace("-", "_")

    ohlcv_json     = json.dumps(chart_data.get("ohlcv", []))
    dates_json     = json.dumps(chart_data.get("dates", []))
    macd_hist_json = json.dumps(chart_data.get("macd_hist", []))
    macd_sig_json  = json.dumps(chart_data.get("macd_signal", []))
    macd_line_json = json.dumps(chart_data.get("macd_line", []))
    rsi_json       = json.dumps(chart_data.get("rsi", []))
    ma5_json       = json.dumps(chart_data.get("ma5", []))
    ma10_json      = json.dumps(chart_data.get("ma10", []))
    ma20_json      = json.dumps(chart_data.get("ma20", []))

    vol_label = "爆量" if vol_ratio >= 2 else "量增" if vol_ratio >= 1.5 else "縮量" if vol_ratio <= 0.5 else "正常"
    vol_color = "#f85149" if vol_ratio >= 2 else "#e3b341" if vol_ratio >= 1.5 else "#8b949e"

    rsi_num = rsi if rsi != "-" else 50
    try:
        rsi_color = _pct_color(float(rsi_num) - 50)
    except Exception:
        rsi_color = "#8b949e"

    return f"""<div class="stock-card" id="card-{stock_id}">
  <div class="card-top">
    <div class="card-id-block">
      <span class="card-code">{stock_id}</span>
      <span class="card-name">{name}</span>
    </div>
    <div class="action-badge" style="color:{action_cfg['color']};background:{action_cfg['bg']};border-color:{action_cfg['color']}">
      {action_cfg['icon']} {action}
    </div>
  </div>

  <div class="price-section">
    <div class="price-main">
      <span class="current-price">{close}</span>
      <span class="price-change" style="color:{_pct_color(change_pct)}">
        {'&#9650;' if change > 0 else '&#9660;'} {abs(change)} ({_pct_str(change_pct)})
      </span>
    </div>
    {f'''<div class="profit-block" style="color:{_profit_color(profit_pct)}">
      持股損益 {_pct_str(profit_pct)} / {'+' if profit_amount >= 0 else ''}{profit_amount:,.0f} 元
      <span class="shares-note">（成本 {cost} x {shares_str} 股）</span>
    </div>''' if shares else ''}
  </div>

  <div class="indicators-row">
    <div class="ind-block">
      <div class="ind-label">RSI(6)</div>
      <div class="ind-value" style="color:{rsi_color}">{rsi}</div>
      <div class="ind-sub">{rsi_status}</div>
    </div>
    <div class="ind-block">
      <div class="ind-label">MACD 柱</div>
      <div class="ind-value" style="color:{_pct_color(macd_hist)}">{'+' if macd_hist > 0 else ''}{macd_hist:.3f}</div>
      <div class="ind-sub">{macd_status}</div>
    </div>
    <div class="ind-block">
      <div class="ind-label">量比</div>
      <div class="ind-value" style="color:{vol_color}">{vol_ratio:.1f}x</div>
      <div class="ind-sub">{vol_label}</div>
    </div>
  </div>

  <div class="indicators-row">
    <div class="ind-block">
      <div class="ind-label">營收年增</div>
      <div class="ind-value" style="color:{_pct_color(rev_yoy) if rev_yoy is not None else '#8b949e'}">{f'{rev_yoy:+.1f}%' if rev_yoy is not None else '-'}</div>
      <div class="ind-sub">{f'月增 {rev_mom:+.1f}%' if rev_mom is not None else '-'}</div>
    </div>
    <div class="ind-block">
      <div class="ind-label">外資</div>
      <div class="ind-value" style="color:{_pct_color(foreign_net) if foreign_net is not None else '#8b949e'}">{f'{foreign_net:+,}張' if foreign_net is not None else '-'}</div>
      <div class="ind-sub">{'買超' if foreign_net and foreign_net > 0 else '賣超' if foreign_net and foreign_net < 0 else '-'}</div>
    </div>
    <div class="ind-block">
      <div class="ind-label">投信</div>
      <div class="ind-value" style="color:{_pct_color(trust_net) if trust_net is not None else '#8b949e'}">{f'{trust_net:+,}張' if trust_net is not None else '-'}</div>
      <div class="ind-sub">{'買超' if trust_net and trust_net > 0 else '賣超' if trust_net and trust_net < 0 else '-'}</div>
    </div>
  </div>

  <div class="mini-charts">
    <div class="mini-chart-wrap">
      <div class="mini-chart-label">K 線（MA5 MA10 MA20）</div>
      <canvas id="kline-{safe_id}" height="100"></canvas>
    </div>
    <div class="mini-chart-wrap">
      <div class="mini-chart-label">成交量</div>
      <canvas id="vol-{safe_id}" height="50"></canvas>
    </div>
    <div class="mini-chart-wrap">
      <div class="mini-chart-label">MACD (6,13,9)</div>
      <canvas id="macd-{safe_id}" height="90"></canvas>
    </div>
    <div class="mini-chart-wrap">
      <div class="mini-chart-label">RSI (14)</div>
      <canvas id="rsi-{safe_id}" height="70"></canvas>
    </div>
  </div>

  <script>
  (function() {{
    var dates = {dates_json};
    var ohlcv = {ohlcv_json};
    var macdH = {macd_hist_json};
    var macdS = {macd_sig_json};
    var macdL = {macd_line_json};
    var rsiD  = {rsi_json};
    var ma5D  = {ma5_json};
    var ma10D = {ma10_json};
    var ma20D = {ma20_json};
    var xAxis = {{ ticks:{{ color:'#8b949e', font:{{size:9,family:'JetBrains Mono'}}, maxTicksLimit:6 }}, grid:{{ color:'rgba(48,54,61,0.5)' }} }};
    var yAxis = {{ ticks:{{ color:'#8b949e', font:{{size:9,family:'JetBrains Mono'}}, maxTicksLimit:4 }}, grid:{{ color:'rgba(48,54,61,0.5)' }}, position:'right' }};
    var noAnim = {{ duration:0 }};

    var kCtx = document.getElementById('kline-{safe_id}');
    if (kCtx && ohlcv.length) {{
      var candleData = ohlcv.map(function(d, i) {{
        return {{ x: i, o: d.o, h: d.h, l: d.l, c: d.c }};
      }});
      var ma5Data  = ma5D.map(function(v, i)  {{ return v !== null ? {{x: i, y: v}} : null; }});
      var ma10Data = ma10D.map(function(v, i) {{ return v !== null ? {{x: i, y: v}} : null; }});
      var ma20Data = ma20D.map(function(v, i) {{ return v !== null ? {{x: i, y: v}} : null; }});
      var kMin = Math.min.apply(null, ohlcv.map(function(d){{return d.l;}}));
      var kMax = Math.max.apply(null, ohlcv.map(function(d){{return d.h;}}));
      var kPad = (kMax - kMin) * 0.08 || kMin * 0.005;
      new Chart(kCtx, {{
        type: 'candlestick',
        data: {{
          datasets: [
            {{
              label: 'K線',
              data: candleData,
              color: {{ up: '#f85149', down: '#3fb950', unchanged: '#8b949e' }},
              borderColor: {{ up: '#f85149', down: '#3fb950', unchanged: '#8b949e' }},
            }},
            {{
              label: 'MA5',
              data: ma5Data,
              type: 'line',
              borderColor: '#e3b341',
              borderWidth: 1.2,
              pointRadius: 0,
              tension: 0.3,
              spanGaps: true,
            }},
            {{
              label: 'MA10',
              data: ma10Data,
              type: 'line',
              borderColor: '#3fb950',
              borderWidth: 1.2,
              pointRadius: 0,
              tension: 0.3,
              spanGaps: true,
            }},
            {{
              label: 'MA20',
              data: ma20Data,
              type: 'line',
              borderColor: '#58a6ff',
              borderWidth: 1.2,
              pointRadius: 0,
              tension: 0.3,
              spanGaps: true,
            }},
          ]
        }},
        options: {{
          animation: noAnim,
          plugins: {{
            legend: {{
              display: true,
              labels: {{
                color: '#8b949e',
                font: {{ size: 9, family: 'JetBrains Mono' }},
                boxWidth: 16,
                padding: 8,
                filter: function(item) {{ return item.text !== 'K線'; }}
              }}
            }},
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  var d = ctx.raw;
                  if (ctx.dataset.label === 'K線') return ['開:'+d.o, '高:'+d.h, '低:'+d.l, '收:'+d.c];
                  return ctx.dataset.label + ': ' + (ctx.parsed.y || '').toFixed(1);
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              type: 'linear',
              ticks: {{
                color: '#8b949e',
                font: {{ size: 9, family: 'JetBrains Mono' }},
                maxTicksLimit: 7,
                callback: function(val) {{
                  var idx = Math.round(val);
                  return (idx >= 0 && idx < dates.length) ? dates[idx] : '';
                }}
              }},
              grid: {{ color: 'rgba(48,54,61,0.5)' }},
              min: -0.5,
              max: dates.length - 0.5,
            }},
            y: {{
              ...yAxis,
              min: kMin - kPad,
              max: kMax + kPad,
            }}
          }}
        }}
      }});
    }}

    var vCtx = document.getElementById('vol-{safe_id}');
    if (vCtx && ohlcv.length) {{
      // 用 sqrt 壓縮量能差距，讓小量天也能看到柱體
      var rawVols = ohlcv.map(function(d){{return d.v;}});
      var volData = ohlcv.map(function(d, i) {{ return {{x: i, y: Math.sqrt(d.v)}}; }});
      var volColors = ohlcv.map(function(d) {{ return d.c >= d.o ? 'rgba(248,81,73,0.7)' : 'rgba(63,185,80,0.7)'; }});
      var sqrtMax = Math.sqrt(Math.max.apply(null, rawVols));
      new Chart(vCtx, {{
        type: 'bar',
        data: {{ datasets: [{{
          data: volData,
          backgroundColor: volColors,
          borderWidth: 0,
          barPercentage: 0.8,
          categoryPercentage: 0.9,
          maxBarThickness: 12,
          minBarLength: 2,
        }}] }},
        options: {{
          animation: noAnim,
          plugins: {{ legend: {{ display: false }}, tooltip: {{
            callbacks: {{
              title: function(ctx) {{ var idx=Math.round(ctx[0].parsed.x); return (idx>=0&&idx<dates.length)?dates[idx]:''; }},
              label: function(ctx) {{
                var idx=Math.round(ctx.parsed.x);
                var real = (idx>=0&&idx<rawVols.length) ? rawVols[idx] : 0;
                return '量: ' + (real/1000).toFixed(0) + ' 張';
              }}
            }}
          }} }},
          scales: {{
            x: {{
              type: 'linear',
              offset: true,
              ticks: {{
                color: '#8b949e',
                font: {{ size: 9, family: 'JetBrains Mono' }},
                maxTicksLimit: 7,
                callback: function(val) {{
                  var idx = Math.round(val);
                  return (idx >= 0 && idx < dates.length) ? dates[idx] : '';
                }}
              }},
              grid: {{ color: 'rgba(48,54,61,0.5)' }},
              min: -0.5,
              max: dates.length - 0.5,
            }},
            y: {{ ...yAxis, min: 0, max: sqrtMax * 1.15,
              ticks: {{ ...yAxis.ticks, maxTicksLimit: 3,
                callback: function(v) {{
                  var real = Math.round(v * v);
                  return real >= 1000 ? (real/1000).toFixed(0)+'K' : real;
                }}
              }}
            }}
          }}
        }}
      }});
    }}

    var mCtx = document.getElementById('macd-{safe_id}');
    if (mCtx && macdH.length) {{
      var macdVals = macdH.concat(macdL).concat(macdS).filter(function(v){{return v!==null;}});
      var macdPad = (Math.max.apply(null,macdVals.map(Math.abs))) * 0.3 || 1;
      var macdHxy = macdH.map(function(v,i){{ return {{x:i, y:v}}; }});
      var macdLxy = macdL.map(function(v,i){{ return v!==null?{{x:i, y:v}}:null; }});
      var macdSxy = macdS.map(function(v,i){{ return v!==null?{{x:i, y:v}}:null; }});
      new Chart(mCtx, {{
        type:'bar',
        data:{{ datasets:[
          {{ data:macdHxy, backgroundColor:macdH.map(function(v){{return v>=0?'rgba(248,81,73,0.7)':'rgba(63,185,80,0.7)'}}), borderWidth:0 }},
          {{ data:macdLxy, type:'line', borderColor:'#58a6ff', borderWidth:1.2, pointRadius:0, tension:0.3, spanGaps:true }},
          {{ data:macdSxy, type:'line', borderColor:'#ffa657', borderWidth:1.2, pointRadius:0, tension:0.3, spanGaps:true }}
        ]}},
        options:{{ animation:noAnim, plugins:{{ legend:{{display:false}} }},
          scales:{{
            x: {{
              type: 'linear',
              ticks: {{
                color: '#8b949e',
                font: {{ size: 9, family: 'JetBrains Mono' }},
                maxTicksLimit: 7,
                callback: function(val) {{
                  var idx = Math.round(val);
                  return (idx >= 0 && idx < dates.length) ? dates[idx] : '';
                }}
              }},
              grid: {{ color: 'rgba(48,54,61,0.5)' }},
              min: -0.5,
              max: dates.length - 0.5,
            }},
            y:{{ ...yAxis,
              min: Math.min.apply(null,macdVals)-macdPad,
              max: Math.max.apply(null,macdVals)+macdPad
            }}
          }}
        }}
      }});
    }}

    var rCtx = document.getElementById('rsi-{safe_id}');
    if (rCtx && rsiD.length) {{
      var rsiVals = rsiD.filter(function(v){{return v!==null;}});
      var rsiMin = Math.max(0,  Math.min.apply(null,rsiVals) - 5);
      var rsiMax = Math.min(100, Math.max.apply(null,rsiVals) + 5);
      var rsiXY = rsiD.map(function(v,i){{ return v!==null?{{x:i,y:v}}:null; }});
      var rsi70 = dates.map(function(_,i){{ return {{x:i,y:70}}; }});
      var rsi30 = dates.map(function(_,i){{ return {{x:i,y:30}}; }});
      var xLinear = {{
        type: 'linear',
        ticks: {{
          color: '#8b949e',
          font: {{ size: 9, family: 'JetBrains Mono' }},
          maxTicksLimit: 7,
          callback: function(val) {{
            var idx = Math.round(val);
            return (idx >= 0 && idx < dates.length) ? dates[idx] : '';
          }}
        }},
        grid: {{ color: 'rgba(48,54,61,0.5)' }},
        min: -0.5,
        max: dates.length - 0.5,
      }};
      new Chart(rCtx, {{
        type:'line',
        data:{{ datasets:[
          {{ data:rsiXY, borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false, spanGaps:true }},
          {{ data:rsi70, borderColor:'rgba(248,81,73,0.35)', borderWidth:1, borderDash:[3,3], pointRadius:0, fill:false }},
          {{ data:rsi30, borderColor:'rgba(63,185,80,0.35)', borderWidth:1, borderDash:[3,3], pointRadius:0, fill:false }}
        ]}},
        options:{{ animation:noAnim, plugins:{{ legend:{{display:false}} }},
          scales:{{ x:xLinear, y:{{ ...yAxis, min:rsiMin, max:rsiMax }} }}
        }}
      }});
    }}
  }})();
  </script>

  <div class="analysis-section">
    <div class="analysis-summary">&#128161; {analysis.get('summary', '')}</div>
    <div class="analysis-details">
      <div class="analysis-item">
        <span class="analysis-label">&#128202; 技術</span>
        <span class="analysis-text">{analysis.get('technical_view', '')}</span>
      </div>
      <div class="analysis-item">
        <span class="analysis-label">&#127974; 籌碼</span>
        <span class="analysis-text">{analysis.get('chip_view', '')}</span>
      </div>
      <div class="analysis-item">
        <span class="analysis-label">&#128200; 基本</span>
        <span class="analysis-text">{analysis.get('fundamental_view', '')}</span>
      </div>
    </div>
    <div class="action-block" style="border-left-color:{action_cfg['color']}">
      <div class="action-title" style="color:{action_cfg['color']}">明日建議：{action}</div>
      <div class="action-text">{analysis.get('action_reason', '')}</div>
      <div class="strategy-row">
        <div class="strategy-item stop-loss">
          <span class="strategy-label">🛑 停損</span>
          <span class="strategy-text">{analysis.get('stop_loss', '—')}</span>
        </div>
        <div class="strategy-item reversal">
          <span class="strategy-label">🔼 多方反轉</span>
          <span class="strategy-text">{analysis.get('reversal_bull', analysis.get('reversal_trigger', '—'))}</span>
        </div>
        <div class="strategy-item reversal">
          <span class="strategy-label">🔽 空方確認</span>
          <span class="strategy-text">{analysis.get('reversal_bear', '—')}</span>
        </div>
      </div>
      <div class="risk-note">&#9888;&#65039; {analysis.get('risk', '')}</div>
    </div>
  </div>

  <div class="card-footer">
    <span class="footer-date">更新：{signals.get('date', '')}</span>
  </div>
</div>"""


def _extract_price_from_text(text):
    """從文字中萃取第一個出現的價位數字"""
    import re
    if not text:
        return "—"
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*元', text)
    if matches:
        return f"{matches[0]} 元"
    return "—"


def _build_strategy_table(all_stock_data):
    """產生作戰總覽表格 HTML"""
    order = {"停損": 0, "減碼": 1, "加碼": 2, "觀望": 3, "持有": 4}
    sorted_data = sorted(all_stock_data, key=lambda s: order.get(s["analysis"].get("action", "觀望"), 5))

    rows = ""
    for s in sorted_data:
        stock_id  = s["stock_id"]
        name      = s["name"]
        signals   = s["signals"]
        analysis  = s["analysis"]
        portfolio = s["portfolio"]

        action     = analysis.get("action", "觀望")
        action_cfg = ACTION_CONFIG.get(action, ACTION_CONFIG["觀望"])
        close      = signals.get("close", 0)
        cost       = portfolio.get("cost_price", 0)
        profit_pct = ((close - cost) / cost * 100) if cost else 0

        ma5  = signals.get("ma5")
        ma20 = signals.get("ma20")
        ma60 = signals.get("ma60")

        supports  = [v for v in [ma5, ma20, ma60] if v and v < close]
        pressures = [v for v in [ma5, ma20, ma60] if v and v > close]
        support_str  = f"{max(supports):.1f}" if supports else "—"
        pressure_str = f"{min(pressures):.1f}" if pressures else "—"

        buy_point = _extract_price_from_text(analysis.get("reversal_trigger", ""))
        if buy_point == "—":
            buy_point = _extract_price_from_text(analysis.get("action_reason", ""))
        stop_price = _extract_price_from_text(analysis.get("stop_loss", ""))

        profit_color = _profit_color(profit_pct)
        profit_str   = f"{'+' if profit_pct >= 0 else ''}{profit_pct:.1f}%"

        rows += f"""<tr class="strategy-row-item">
  <td class="st-stock"><span class="st-code">{stock_id}</span><span class="st-name">{name}</span></td>
  <td data-label="收盤" class="st-close">{close}</td>
  <td data-label="損益" class="st-profit" style="color:{profit_color}">{profit_str}</td>
  <td data-label="操作" class="st-action"><span class="st-badge" style="color:{action_cfg['color']};background:{action_cfg['bg']};border-color:{action_cfg['color']}">{action_cfg['icon']} {action}</span></td>
  <td data-label="支撐" class="st-support">{support_str}</td>
  <td data-label="壓力" class="st-pressure">{pressure_str}</td>
  <td data-label="買點參考" class="st-buy">{buy_point}</td>
  <td data-label="停損位" class="st-stop">{stop_price}</td>
  <td data-label="操作理由" class="st-reason">{analysis.get('action_reason', '—')}</td>
</tr>"""

    return f"""<div class="strategy-overview">
  <div class="strategy-overview-title" onclick="toggleStrategyTable()" style="cursor:pointer;user-select:none;">
    ⚔️ 作戰總覽
    <span class="strategy-toggle-icon" id="strategy-toggle-icon">▲</span>
  </div>
  <div class="strategy-table-wrap" id="strategy-table-wrap">
    <table class="strategy-table">
      <thead>
        <tr>
          <th>股票</th><th>收盤</th><th>損益</th><th>操作</th>
          <th>支撐</th><th>壓力</th><th>買點參考</th><th>停損位</th><th>操作理由</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
<script>
function toggleStrategyTable() {{
  var wrap = document.getElementById('strategy-table-wrap');
  var icon = document.getElementById('strategy-toggle-icon');
  if (wrap.style.display === 'none') {{ wrap.style.display = ''; icon.textContent = '▲'; }}
  else {{ wrap.style.display = 'none'; icon.textContent = '▼'; }}
}}
</script>"""


def generate_report(all_stock_data, report_date=None):
    if not report_date:
        report_date = datetime.today().strftime("%Y/%m/%d")

    total_stocks = len(all_stock_data)
    up_count     = sum(1 for s in all_stock_data if s["signals"].get("change_pct", 0) > 0)
    down_count   = sum(1 for s in all_stock_data if s["signals"].get("change_pct", 0) < 0)
    flat_count   = total_stocks - up_count - down_count

    action_counts = {}
    for s in all_stock_data:
        a = s["analysis"].get("action", "觀望")
        action_counts[a] = action_counts.get(a, 0) + 1

    total_profit = sum(
        (s["signals"].get("close", 0) - s["portfolio"].get("cost_price", 0))
        * s["portfolio"].get("shares", 0)
        for s in all_stock_data
    )

    total_market_value = sum(
        s["signals"].get("close", 0) * s["portfolio"].get("shares", 0)
        for s in all_stock_data
    )
    total_cost = sum(
        s["portfolio"].get("cost_price", 0) * s["portfolio"].get("shares", 0)
        for s in all_stock_data
    )
    total_return_pct = (total_profit / total_cost * 100) if total_cost else 0
    avg_vol_ratio = (
        sum(s["signals"].get("vol_ratio", 1) or 1 for s in all_stock_data) / total_stocks
        if total_stocks else 1
    )
    action_counts_html = "".join([
        f'<div class="stat-item">'
        f'<span class="label">{a}</span>'
        f'<span class="value" style="color:{ACTION_CONFIG.get(a, ACTION_CONFIG["觀望"])["color"]}">{c}</span>'
        f'</div>'
        for a, c in action_counts.items()
    ])

    action_summary_items = ""
    for action in ["加碼", "停損", "減碼"]:
        stocks_with = [s for s in all_stock_data if s["analysis"].get("action") == action]
        if stocks_with:
            cfg = ACTION_CONFIG[action]
            names = "、".join([f"{s['stock_id']} {s['name']}" for s in stocks_with])
            action_summary_items += (
                f'<div class="action-summary-item">'
                f'<span style="color:{cfg["color"]};font-weight:600">{cfg["icon"]} {action}</span>'
                f'<span class="action-summary-stocks">{names}</span>'
                f'</div>'
            )

    strategy_table_html = _build_strategy_table(all_stock_data)

    cards_html = "\n".join([generate_stock_card(s) for s in all_stock_data])
    profit_color = "var(--red)" if total_profit >= 0 else "var(--green)"
    profit_sign  = "+" if total_profit >= 0 else ""
    return_color = "var(--red)" if total_return_pct >= 0 else "var(--green)"
    return_sign  = "+" if total_return_pct >= 0 else ""
    vol_color    = "#f85149" if avg_vol_ratio >= 2 else "#e3b341" if avg_vol_ratio >= 1.5 else "#8b949e"

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>台股覆盤報告 {report_date}</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1/dist/chartjs-chart-financial.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
<style>
:root {{
  --bg:#0d1117; --bg2:#161b22; --bg3:#1c2128; --border:#30363d;
  --text:#e6edf3; --text2:#8b949e;
  --green:#3fb950; --red:#f85149; --blue:#58a6ff; --yellow:#e3b341; --orange:#ffa657;
}}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{ background:var(--bg); color:var(--text); font-family:'Noto Sans TC',sans-serif; min-height:100vh; line-height:1.6; }}
.report-header{{ background:var(--bg2); border-bottom:1px solid var(--border); padding:24px 32px; }}
.header-top{{ display:flex; align-items:center; gap:16px; margin-bottom:20px; }}
.report-title{{ font-size:20px; font-weight:700; letter-spacing:1px; }}
.report-date{{ font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--text2); background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:4px 12px; margin-left:auto; }}
.stats-bar{{ display:flex; gap:24px; flex-wrap:wrap; align-items:center; }}
.stat-item{{ display:flex; flex-direction:column; gap:2px; }}
.stat-item .label{{ font-size:11px; color:var(--text2); text-transform:uppercase; letter-spacing:.5px; }}
.stat-item .value{{ font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:600; }}
.divider{{ width:1px; height:36px; background:var(--border); }}
.action-summary{{ background:var(--bg3); border:1px solid var(--border); border-radius:10px; padding:14px 20px; margin:20px 32px 0; display:flex; gap:20px; flex-wrap:wrap; align-items:center; }}
.action-summary-title{{ font-size:12px; color:var(--text2); font-weight:500; text-transform:uppercase; white-space:nowrap; }}
.action-summary-item{{ display:flex; align-items:center; gap:8px; font-size:13px; }}
.action-summary-stocks{{ color:var(--text2); font-size:12px; }}
.main-grid{{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:16px; padding:24px 32px 48px; }}
.stock-card{{ background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:20px; display:flex; flex-direction:column; gap:14px; transition:border-color .2s,transform .2s; animation:fadeIn .4s ease both; }}
.stock-card:hover{{ border-color:#58a6ff44; transform:translateY(-2px); }}
@keyframes fadeIn{{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
.card-top{{ display:flex; align-items:center; justify-content:space-between; }}
.card-id-block{{ display:flex; align-items:baseline; gap:8px; }}
.card-code{{ font-family:'JetBrains Mono',monospace; font-size:18px; font-weight:600; color:var(--blue); }}
.card-name{{ font-size:15px; font-weight:500; }}
.action-badge{{ border:1px solid; border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; }}
.price-section{{ display:flex; flex-direction:column; gap:4px; padding-bottom:14px; border-bottom:1px solid var(--border); }}
.price-main{{ display:flex; align-items:baseline; gap:10px; }}
.current-price{{ font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:600; }}
.price-change{{ font-family:'JetBrains Mono',monospace; font-size:14px; }}
.profit-block{{ font-family:'JetBrains Mono',monospace; font-size:13px; }}
.shares-note{{ font-size:11px; color:var(--text2); margin-left:6px; }}
.indicators-row{{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
.ind-block{{ background:var(--bg3); border-radius:8px; padding:8px; text-align:center; }}
.ind-label{{ font-size:10px; color:var(--text2); margin-bottom:4px; }}
.ind-value{{ font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:600; line-height:1.2; }}
.ind-sub{{ font-size:10px; color:var(--text2); margin-top:2px; }}
.mini-charts{{ display:flex; flex-direction:column; gap:6px; background:var(--bg3); border-radius:10px; padding:12px; }}
.mini-chart-wrap{{ display:flex; flex-direction:column; gap:3px; }}
.mini-chart-label{{ font-size:10px; color:var(--text2); letter-spacing:.3px; }}
canvas{{ width:100% !important; }}
.analysis-section{{ display:flex; flex-direction:column; gap:10px; }}
.analysis-summary{{ font-size:13px; font-weight:500; background:var(--bg3); border-radius:8px; padding:10px 12px; line-height:1.5; }}
.analysis-details{{ display:flex; flex-direction:column; gap:6px; }}
.analysis-item{{ display:flex; gap:8px; font-size:12px; line-height:1.5; }}
.analysis-label{{ color:var(--text2); white-space:nowrap; flex-shrink:0; font-size:11px; margin-top:1px; }}
.action-block{{ border-left:3px solid; padding:10px 14px; background:var(--bg3); border-radius:0 8px 8px 0; display:flex; flex-direction:column; gap:6px; }}
.action-title{{ font-size:13px; font-weight:700; }}
.action-text{{ font-size:12px; line-height:1.5; }}
.strategy-row{{ display:flex; flex-direction:column; gap:5px; margin-top:2px; padding-top:6px; border-top:1px solid var(--border); }}
.strategy-item{{ display:flex; gap:6px; font-size:11px; line-height:1.5; }}
.strategy-label{{ white-space:nowrap; flex-shrink:0; font-weight:600; }}
.stop-loss .strategy-label{{ color:#f85149; }}
.reversal .strategy-label{{ color:#3fb950; }}
.strategy-text{{ color:var(--text2); }}
.risk-note{{ font-size:11px; color:var(--text2); }}
.card-footer{{ display:flex; align-items:center; justify-content:flex-end; padding-top:10px; border-top:1px solid var(--border); }}
.footer-date{{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text2); }}
/* ── 作戰總覽 ── */
.strategy-overview{{ margin:16px 32px 0; background:var(--bg3); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.strategy-overview-title{{ padding:12px 20px; font-size:13px; font-weight:600; color:var(--text); border-bottom:1px solid var(--border); letter-spacing:.5px; display:flex; justify-content:space-between; align-items:center; transition:background .15s; }}
.strategy-overview-title:hover{{ background:rgba(88,166,255,0.05); }}
.strategy-toggle-icon{{ font-size:11px; color:var(--text2); }}
.strategy-table-wrap{{ overflow-x:auto; }}
.strategy-table{{ width:100%; border-collapse:collapse; font-size:12px; }}
.strategy-table thead tr{{ background:var(--bg2); }}
.strategy-table th{{ padding:8px 14px; text-align:left; color:var(--text2); font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:.4px; white-space:nowrap; border-bottom:1px solid var(--border); }}
.strategy-row-item{{ border-bottom:1px solid rgba(48,54,61,0.6); transition:background .15s; }}
.strategy-row-item:last-child{{ border-bottom:none; }}
.strategy-row-item:hover{{ background:rgba(88,166,255,0.04); }}
.strategy-table td{{ padding:9px 14px; vertical-align:middle; }}
.st-stock{{ display:flex; flex-direction:column; gap:2px; white-space:nowrap; }}
.st-code{{ font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:600; color:var(--blue); }}
.st-name{{ font-size:11px; color:var(--text2); }}
.st-close{{ font-family:'JetBrains Mono',monospace; font-weight:600; white-space:nowrap; }}
.st-profit{{ font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; white-space:nowrap; }}
.st-badge{{ border:1px solid; border-radius:20px; padding:2px 10px; font-size:11px; font-weight:600; white-space:nowrap; }}
.st-support{{ font-family:'JetBrains Mono',monospace; font-size:12px; color:#3fb950; white-space:nowrap; }}
.st-pressure{{ font-family:'JetBrains Mono',monospace; font-size:12px; color:#f85149; white-space:nowrap; }}
.st-buy{{ font-family:'JetBrains Mono',monospace; font-size:12px; color:#e3b341; white-space:nowrap; }}
.st-stop{{ font-family:'JetBrains Mono',monospace; font-size:12px; color:#ffa657; white-space:nowrap; }}
.st-reason{{ color:var(--text2); font-size:11px; line-height:1.5; max-width:320px; }}

/* ── 手機版：卡片式排版 ── */
@media(max-width:600px){{
  .main-grid{{ grid-template-columns:1fr; padding:16px; }}
  .report-header{{ padding:16px; }}
  .action-summary{{ margin:12px 16px 0; }}
  .strategy-overview{{ margin:12px 16px 0; }}
  .indicators-row{{ grid-template-columns:repeat(2,1fr); }}
  .strategy-table thead{{ display:none; }}
  .strategy-table, .strategy-table tbody, .strategy-table tr, .strategy-table td{{ display:block; width:100%; }}
  .strategy-row-item{{ padding:12px 16px; border-bottom:1px solid var(--border); }}
  .strategy-row-item td{{ padding:0; border:none; }}
  .strategy-row-item td::before{{
    content: attr(data-label);
    font-size:10px; color:var(--text2); text-transform:uppercase;
    letter-spacing:.4px; display:block; margin-bottom:2px; margin-top:6px;
  }}
  .strategy-row-item td:first-child{{ margin-top:0; }}
  .strategy-row-item td:first-child::before{{ display:none; }}
  .st-stock{{ flex-direction:row; align-items:center; gap:8px; margin-bottom:8px; }}
  .st-name{{ font-size:13px; color:var(--text); }}
  .st-reason{{ max-width:100%; }}
}}
</style>
</head>
<body>
<div class="report-header">
  <div class="header-top">
    <span class="report-title">&#128202; 台股盤後覆盤報告</span>
    <span class="report-date">{report_date} 收盤後</span>
  </div>
  <div class="stats-bar">
    <div class="stat-item"><span class="label">持股總數</span><span class="value" style="color:var(--blue)">{total_stocks}</span></div>
    <div class="divider"></div>
    <div class="stat-item"><span class="label">上漲</span><span class="value" style="color:var(--red)">{up_count}</span></div>
    <div class="stat-item"><span class="label">持平</span><span class="value" style="color:var(--text2)">{flat_count}</span></div>
    <div class="stat-item"><span class="label">下跌</span><span class="value" style="color:var(--green)">{down_count}</span></div>
    <div class="divider"></div>
    <div class="stat-item">
      <span class="label">總市值</span>
      <span class="value" style="color:var(--text)">{total_market_value:,.0f} 元</span>
    </div>
    <div class="stat-item">
      <span class="label">總成本</span>
      <span class="value" style="color:var(--text2)">{total_cost:,.0f} 元</span>
    </div>
    <div class="divider"></div>
    <div class="stat-item">
      <span class="label">持股浮動損益</span>
      <span class="value" style="color:{profit_color}">{profit_sign}{total_profit:,.0f} 元</span>
    </div>
    <div class="stat-item">
      <span class="label">整體報酬率</span>
      <span class="value" style="color:{return_color}">{return_sign}{total_return_pct:.2f}%</span>
    </div>
    <div class="divider"></div>
    <div class="stat-item">
      <span class="label">平均量比</span>
      <span class="value" style="color:{vol_color}">{avg_vol_ratio:.2f}x</span>
    </div>
    <div class="divider"></div>
    {action_counts_html}
  </div>
</div>
{f'<div class="action-summary"><span class="action-summary-title">&#9889; 今日建議</span>{action_summary_items}</div>' if action_summary_items else ''}
{strategy_table_html}
<div class="main-grid">
{cards_html}
</div>
<script>
var order={{'停損':0,'減碼':1,'加碼':2,'觀望':3,'持有':4}};
var grid=document.querySelector('.main-grid');
var cards=Array.from(grid.querySelectorAll('.stock-card'));
cards.sort(function(a,b){{
  function getA(el){{return el.querySelector('.action-badge').textContent.trim().replace(/[^\u4e00-\u9fff]/g,'').trim();}}
  return (order[getA(a)]||5)-(order[getA(b)]||5);
}});
cards.forEach(function(c,i){{c.style.animationDelay=i*0.05+'s';grid.appendChild(c);}});
</script>
</body>
</html>"""
