# this is only for generate the plotly graph.
# 这个代码中有大量细节的参数，基本可用于之后的plotly 配置到gitHub pages上。
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import textwrap # 用来给悬浮窗自动换行
# 以下用来提供下载交互式图表
from base64 import b64encode
import io

buffer = io.StringIO()

# 读取数据
dataset = pd.read_csv('../../githubdata/ScientistsWithDatasets-1data/dataset.csv', index_col=0)
author = pd.read_csv('../../githubdata/ScientistsWithDatasets-1data/author.csv', index_col=0)
B2AIauthor = author.loc[author['isB2AI'] == 1]

GreatGPT = ['Bioentities: Cancer Signaling Molecules<br>Keywords: Therapeutic Mechanisms', 'Bioentities: Health & Disease Risk Factors<br>Keywords: Epidemiology & Research Methods',
'Bioentities: Breast Cancer Genetics Biomarkers<br>Keywords: Genetics & Biomarkers', 
'Bioentities: Breast Cancer Treatments<br>Keywords: Cancer Treatment & Management',
'Bioentities: Tumor Microenvironment Metastasis Molecules<br>Keywords: Tumor Metastasis & Matrix Biology']
ClusterPosition = [[-10,50],[55,-50],[-40,-50],[75,20],[-75,20],]
# Clusteranlge = [0,-45,0,45,-60]
# 开始画图，构建一个Figure()
figSwD = go.Figure()
# 5个聚类的trace先画出来，C0author只是个名字，代表这一类的作者
for i in range(5):
    C0author = author.loc[(author['clusterID'] == i) & (author['isB2AI'] == 0)]
    # Scattergl专为大型网络打造，比一般的scatter更流畅
    figSwD.add_trace(go.Scattergl(x=C0author['X'], y=C0author['Y'],
                            #设置一些自定义数据，hover时后展示
                            customdata=np.stack((C0author.index, C0author['Name'],C0author['MainAffi'].apply(lambda t: str(t)).apply(lambda t: "<br>".join(textwrap.wrap(t,40)) # textwrap是一个自动换行的好方法
), C0author['clusterID'], C0author['CareerAge'], C0author['CitedNum'],  C0author['isB2AI']), axis=-1),
                            mode='markers',
        hovertemplate='<b>Author ID<b>: %{customdata[0]}<br><b>Author Name<b>: %{customdata[1]}<br><b>Cluster ID<b>: %{customdata[3]}<br><b>Career Age<b>: %{customdata[4]}<br><b>#Cited<b>: %{customdata[5]}<br><b>isB2AI<b>: %{customdata[6]}<br><br><b>Author Main Affiliation:<br><b> %{customdata[2]}<extra></extra>',
        # <extra></extra> 是为了隐藏第二个框框，不加这行代码就很丑
                              marker=dict(size=3),# 设置了较小的值，代表一般作者的大小
                            # this is the name of the trace
                            name=GreatGPT[i],
                            ))
# 此处是dataset的trace
figSwD.add_trace(go.Scattergl(x=dataset['X'], y=dataset['Y'],
                              marker=dict(size=10, color='black'),# 深色代表数据，更合适
                              # 方框
                              marker_symbol = 31,
    customdata=np.stack((dataset.index, dataset['Name']), axis=-1),
    mode='markers',
    hovertemplate='<br><br><b>Dataset ID<b>: %{customdata[0]}<br><b>Dataset Name<b>: %{customdata[1]}<extra></extra>',
    name='<b>'+'DataSets'+'<b>',
   
))
# B2AIauthor，breast cancer area只有六个人
figSwD.add_trace(go.Scattergl(x=B2AIauthor['X'], y=B2AIauthor['Y'],
                              customdata=np.stack((B2AIauthor.index, B2AIauthor['Name'], B2AIauthor['MainAffi'].apply(lambda t: str(t)).apply(lambda t: "<br>".join(textwrap.wrap(t, 30))
                                                                                                                                        ), B2AIauthor['clusterID'], B2AIauthor['CareerAge'], B2AIauthor['CitedNum'],  B2AIauthor['isB2AI']), axis=-1),
                              mode='markers',
                              hovertemplate='<b>Author ID<b>: %{customdata[0]}<br><b>Author Name<b>: %{customdata[1]}<br><b>Cluster ID<b>: %{customdata[3]}<br><b>Career Age<b>: %{customdata[4]}<br><b>#Cited<b>: %{customdata[5]}<br><b>isB2AI<b>: %{customdata[6]}<br><br><b>Author Main Affiliation<br><b>: %{customdata[2]}<extra></extra>',
                              marker=dict(size=30),
                              marker_symbol=22,
                              # this is the name of the trace
                              name='<b>'+'Bridge2AI Talent in Breast Cancer' + '<b>',
                              ))
for i in range(5):
    figSwD.add_annotation(
    x=ClusterPosition[i][0],
    y=ClusterPosition[i][1],
    xref="x",
    yref="y",
    text='<b>'+GreatGPT[i]+'<b>',
        showarrow=False,
    font=dict(
        size=10,
        color=px.colors.qualitative.Plotly[i]
    ),
    align="center",
        bordercolor="#FFFFFF",
        borderwidth=2,
        borderpad=4,
        bgcolor="#FFFFFF",
    opacity=0.95,
        # textangle=Clusteranlge[i]
)

# 设置下整体的layout，注意给legend 设置的位置; 以及配色。
figSwD.update_scenes(camera_center=dict(x=0.5,y=0,z=0),)
figSwD.update_layout(
    title='Breast Cancer Area: Scientists with Datasets <br> <br> <sup>Hover over a node to see <b> details</b>.  Use the mouse wheel or the ‘+’ and ‘-’ buttons on the top right of the canvas to <b>zoom in and out </b>.</sup>',
    xaxis_range=[-50,150],
    plot_bgcolor='rgba(229 , 236, 264, 1)',
    # paper_bgcolor='#fff',
    legend=dict(itemsizing='constant',title_font_family="Times New Roman",
                              font=dict(size= 12),
                              yanchor="top",  xanchor="right",x=0.99,y=0.98,
                              bgcolor = 'rgba(255 , 255, 255, 0.9)'),
    dragmode='pan',
    hoverlabel_align = 'left',
    hoverlabel=dict(
        font_size=10,
    ),
    yaxis_visible=False, 
    height=800,
                     yaxis_showticklabels=False, 
                     xaxis_visible=False, 
                     xaxis_showticklabels=False,
                     
    yaxis={
                        'scaleanchor': 'x',
                        'scaleratio': 1
                    },
                     )

# 这里把图表写了下来，可以方便的分享，保留交互功能
figSwD.write_html(buffer, config={'scrollZoom': True, 'responsive': True,})
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

# 一些样式表
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

# 上边是绘制好了 plotly的图表，下面用dash，生成网页
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,)
app.title = "ScientistWithDataset"
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="🔬", className="header-emoji"),
                html.H1(
                    children="Scientists with datasets", className="header-title"
                ),
            ],
            className="header",
        ),

        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="MDS", # ID随便起的，似乎不重要
                        figure=figSwD,# 这里很重要，是上边画的图
                        config={'scrollZoom': True,
                                'responsive': True,
                                },  # 鼠标滚轴可以缩放，默认关闭，这里打开
                    ),
                    className="card",
                ),
            ],

            className="wrapper",
        ),
        # 提供一个下载链接，可以将图表保存为静态网页，比较方便。
        html.A(
            html.Button("Download as HTML"),
            id="download",
            href="data:text/html;base64," + encoded,
            download="plotly_graph.html"
        )
    ]
    
)

if __name__ == "__main__":
    app.run_server(debug=True)
