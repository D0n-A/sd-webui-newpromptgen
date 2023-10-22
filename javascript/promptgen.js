function promptGenSend(text, where){
    const textarea = gradioApp().querySelector('#promptgen_selected_text textarea')
    textarea.value = text
    updateInput(textarea)
    gradioApp().querySelector('#promptgen_send_to_'+where).click()
    where == 'txt2img' ? switch_to_txt2img() : switch_to_img2img()
}

function promptGenSubmit(){
    const id = randomId()
    requestProgress(id, gradioApp().getElementById('promptgen_results_column'), null, function(){})
    const res = create_submit_args(arguments)
    res[0] = id
    return res
}
