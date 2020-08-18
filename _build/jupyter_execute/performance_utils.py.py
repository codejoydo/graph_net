## Confusion matrix

predictions = tf.concat(predictions, axis=0)
predictions = tf.argmax(predictions, axis=1, output_type=tf.int32)

labels = tf.argmax(test_y, axis=1, output_type=tf.int32)

confusion_matrix = tf.math.confusion_matrix(labels, 
                                            predictions, 
                                            num_classes=num_clips, 
                                            dtype=tf.float32)
print(np.sum(np.sum(confusion_matrix, axis=1))/76)
print(confusion_matrix)

import plotly.figure_factory as ff

# predicted clip names
clip_names_pred = clip_names
# true clip names
clip_names_true =  clip_names

# change each element of z to type string for annotations
confusion_matrix_text = [[str(y) for y in x] for x in confusion_matrix]

# set up figure 
fig = ff.create_annotated_heatmap(confusion_matrix, 
                                  x=clip_names_pred, 
                                  y=clip_names_true, 
                                  annotation_text=confusion_matrix_text, 
                                  colorscale='Viridis')

# add title
fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted clip names",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.35,
                        y=0.5,
                        showarrow=False,
                        text="True clip names",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))

# add colorbar
fig['data'][0]['showscale'] = True
fig.show()