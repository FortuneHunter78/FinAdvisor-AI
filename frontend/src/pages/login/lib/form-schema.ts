import type { HTMLInputTypeAttribute } from 'react';
import { z } from 'zod';
import { validationMessages } from './form.messages';

export const LoginFormSchema = z.object({
  username: z
    .string({ message: validationMessages.requared })
    .min(3, { message: validationMessages.length }),
  password: z
    .string({ message: validationMessages.requared })
    .min(3, { message: validationMessages.length }),
});

export type loginFormSchemaType = z.infer<typeof LoginFormSchema>;

interface IFormField {
  name: keyof loginFormSchemaType;
  label: string;
  type?: HTMLInputTypeAttribute;
}

export const loginFormFields: IFormField[] = [
  { name: 'username', label: 'Имя пользователя' },
  { name: 'password', label: 'Пароль', type: 'password' },
];
